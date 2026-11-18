% Time NWB reads and validate a CPU baseline on labeled trials.
clearvars
rng(42);
nFold = 5;
lambda = 0.10;

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
[ok, dirInfo] = fileattrib(tempdir);
assert(ok, 'Cannot find the temporary folder.');
tmp = tempname(dirInfo.Name);
assert(strcmpi(fileparts(tmp), dirInfo.Name), 'Unexpected extraction path.');
mkdir(tmp);
cleanup = onCleanup(@() rmdir(tmp, 's'));

t = tic;
unzip(fullfile(root, 'player_data.zip'), tmp);
fprintf('Extract: %.2f s\n', toc(t));

files = dir(fullfile(tmp, 'player_data', 'challenge*', '*.nwb'));
trials = '/intervals/trials';
rates = '/processing/ecephys/Firing_rate_1000ms/data';
tRead = tic;
for k = 1:numel(files)
    fn = fullfile(files(k).folder, files(k).name);
    t = tic;
    id = h5read(fn, [trials '/id']);
    units = h5read(fn, '/units/id');
    fr = h5read(fn, rates);
    [~, group] = fileparts(files(k).folder);
    isLabeled = endsWith(fn, {'-train.nwb', '-eval-1.nwb'});
    fprintf('%s %s: %d trials, %d units, %s rates (%s), %.2f s\n', ...
        group, files(k).name, numel(id), numel(units), ...
        mat2str(size(fr)), class(fr), toc(t));
    delay = h5read(fn, [trials '/delay_duration']);
    data(k).x = trialMeans(fr, id, delay);
    data(k).y = [];
    data(k).delay = delay(:);
    if isLabeled
        cue = h5read(fn, [trials '/sample_cue']);
        correct = h5read(fn, [trials '/is_correct']);
        if iscell(correct) || isstring(correct)
            assert(all(ismember(upper(string(correct)), ["TRUE", "FALSE"])), ...
                'Unexpected behavior label.');
            correct = strcmpi(correct, 'TRUE');
        end
        assert(all(ismember(cue, [4 8])) && all(ismember(correct, [0 1])), ...
            'Unexpected label value.');
        data(k).y = [cue(:) == 8, correct(:) == 1];
    end
end
fprintf('Read and extract trial features: %.2f s\n', toc(tRead));

% Hold out whole trials. For eval-1, estimate what the small calibration set can do.
for k = 1:numel(files)
    name = files(k).name;
    isCh1Train = endsWith(files(k).folder, 'challenge1') && endsWith(name, '-train.nwb');
    if ~isCh1Train && ~endsWith(name, '-eval-1.nwb')
        continue
    end
    d = data(k);
    t = tic;
    groups = unique([d.delay, d.y], 'rows');
    fold = zeros(size(d.delay));
    offset = 0;
    for j = 1:size(groups, 1)
        rows = find(all([d.delay, d.y] == groups(j, :), 2));
        rows = rows(randperm(numel(rows)));
        fold(rows) = mod(offset + (0:numel(rows)-1), nFold) + 1;
        offset = offset + numel(rows);
    end
    pred = false(size(d.y));
    for j = 1:nFold
        val = fold == j;
        pred(val, :) = ridgeClass(d.x(~val, :), d.y(~val, :), d.x(val, :), lambda);
    end
    acc = mean(pred == d.y, 1);
    bacc = (sum(pred & d.y, 1)./sum(d.y, 1) + ...
        sum(~pred & ~d.y, 1)./sum(~d.y, 1)) / 2;
    fprintf('%s: accuracy %.3f / %.3f, balanced accuracy %.3f / %.3f, %.2f s\n', ...
        name, acc, bacc, toc(t));
end

clear cleanup

function x = trialMeans(fr, id, delay)
x = zeros(numel(id), size(fr, 1) - 2);
for i = 1:numel(id)
    cols = fr(1, :) == id(i);
    assert(nnz(cols) == delay(i), 'Rate bins and trial duration do not match.');
    x(i, :) = mean(fr(3:end, cols), 2).';
end
end

function pred = ridgeClass(x, y, xv, lambda)
% Scale from the training fold only. Fit both binary outcomes at once.
center = mean(x, 1);
scale = max(std(x, 0, 1), 1e-6);
keep = scale > 1e-6;
x = (x(:, keep) - center(keep)) ./ scale(keep);
xv = (xv(:, keep) - center(keep)) ./ scale(keep);
p = size(x, 2);
y = double(y);
bias = mean(y, 1);
a = (x * x.' / p + lambda * eye(size(x, 1))) \ (y - bias);
pred = xv * x.' / p * a + bias >= 0.5;
end
