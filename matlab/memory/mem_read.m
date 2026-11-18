function d = mem_read(pack)
% Read delay-period spikes and labels from the competition archive.
[ok, base] = fileattrib(tempdir);
assert(ok, 'Temporary directory is unavailable.');
tmp = tempname(base.Name);
assert(strcmpi(fileparts(tmp), base.Name), 'Unexpected extraction directory.');
mkdir(tmp);
clean = onCleanup(@() rmdir(tmp, 's')); %#ok<NASGU>
unzip(pack, tmp);

files = dir(fullfile(tmp, 'player_data', 'challenge*', '*.nwb'));
assert(numel(files) == 13, 'Expected the 13 official memory NWB files.');
for k = 1:numel(files)
    t = tic;
    fn = fullfile(files(k).folder, files(k).name);
    [~, grp] = fileparts(files(k).folder);
    d(k) = readFile(fn, files(k).name, str2double(grp(end))); %#ok<AGROW>
    fprintf('Read %s: %d trials, %d units, %.2f s\n', ...
        d(k).name, numel(d(k).id), size(d(k).f, 2), toc(t));
end
end

function d = readFile(fn, name, ch)
tr = '/intervals/trials/';
u = '/units/';
d.name = name;
d.ch = ch;
d.id = double(h5read(fn, [tr 'id']));
d.delay = double(h5read(fn, [tr 'delay_duration']));
assert(all(ismember(d.delay, [3 6])) && numel(unique(d.id)) == numel(d.id));
d.units = [double(h5read(fn, [u 'id'])), double(h5read(fn, [u 'Cluster_id'])), ...
    double(h5read(fn, [u 'shank'])), double(h5read(fn, [u 'channel']))];
d.reg = strings(size(d.units, 1), 3);
for k = 1:3
    lv = [4 5 6];
    d.reg(:, k) = string(h5read(fn, [u 'anatomic_level_' num2str(lv(k))]));
end
d.y = [];
if endsWith(name, {'-train.nwb', '-eval-1.nwb'})
    cue = h5read(fn, [tr 'sample_cue']);
    cor = h5read(fn, [tr 'is_correct']);
    if iscell(cor) || isstring(cor)
        cor = upper(string(cor));
        assert(all(ismember(cor, ["TRUE", "FALSE"])));
        cor = cor == "TRUE";
    end
    assert(all(ismember(cue, [4 8])) && all(ismember(cor, [0 1])));
    d.y = [cue(:) == 8, cor(:) == 1];
end

sp = h5read(fn, [u 'spike_times']);
idx = double(h5read(fn, [u 'spike_times_index']));
assert(size(sp, 1) == 2 && idx(end) == size(sp, 2) && numel(idx) == size(d.units, 1));
[d.f, d.pop, d.dyn] = spikeFeatures(sp, idx, d.id, d.delay);
end

function [f, pop, dyn] = spikeFeatures(sp, idx, id, delay)
% Seconds from delay onset. Negative windows are relative to delay end.
% Columns 1-7 cover the full delay and six early cumulative windows.
win = [0 Inf; 0 .05; 0 .10; 0 .25; 0 .50; 0 1; 0 1.5; ...
    .05 .15; .15 .30; .30 .55; .55 .90; .90 1.35; 1.35 2; 1 2.5; ...
    -1.5 -.75; -.75 -.25; -.25 0; -1 0; -.50 0; -.10 0; ...
    0 1; 1 2; 2 3];
bin = .05;
n = numel(id);
nb = round(max(delay) / bin);
lo = repmat(win(:, 1).', n, 1);
hi = repmat(win(:, 2).', n, 1);
tail = win(:, 1) < 0;
lo(:, tail) = lo(:, tail) + delay;
hi(:, tail) = hi(:, tail) + delay;
hi = min(hi, delay);
% Windows 24:26 partition either delay into three nonoverlapping parts.
lo = [lo, delay .* [0 1/3 2/3]];
hi = [hi, delay .* [1/3 2/3 1]];
width = hi - lo;
row = (1:n).';
lo = row + n * round(lo / bin);
hi = row + n * round(hi / bin);

nu = numel(idx);
f = zeros(n, nu, size(width, 2), 'single');
total = zeros(n, nb, 'single');
active = zeros(n, nb, 'single');
byId = zeros(max(id) + 1, 1);
byId(id + 1) = row;
ix = [0; idx(:)];
for j = 1:nu
    a = ix(j) + 1 : ix(j + 1);
    r = byId(sp(1, a) + 1);
    assert(all(r > 0), 'Unexpected trial ID in a spike record.');
    st = sp(2, a).';
    keep = st >= 0 & st < delay(r);
    b = floor(st(keep) / bin);
    cnt = reshape(accumarray(r(keep) + n * b, 1, [n * nb, 1]), n, nb);
    cs = [zeros(n, 1), cumsum(cnt, 2)];
    f(:, j, :) = reshape(single((cs(hi) - cs(lo)) ./ width), n, 1, []);
    total = total + single(cnt);
    active = active + single(cnt > 0);
end

% Population bursts in the first and last second, and throughout the delay.
pop = zeros(n, 12, 'single');
dyn = zeros(n, 18, 'single');
for i = 1:n
    last = round(delay(i) / bin);
    part = {1:20, last-19:last, 1:last};
    for j = 1:3
        x = total(i, part{j}) / nu;
        a = active(i, part{j}) / nu;
        pop(i, (j-1)*4+(1:4)) = [mean(x), std(x), max(x), mean(a)];
    end
    % Rate changes and oscillations of the population within the delay.
    part = {1:20, 21:last-20, last-19:last};
    allMean = mean(total(i, 1:last)) / nu;
    for j = 1:3
        x = double(total(i, part{j})) / nu;
        a = double(active(i, part{j})) / nu;
        p = abs(fft(x-mean(x))).^2;
        freq = (0:numel(x)-1) / (numel(x)*bin);
        p = p(freq > 0 & freq <= 10);
        freq = freq(freq > 0 & freq <= 10);
        fast = sum(p(freq >= 4)) / max(sum(p), eps);
        ratio = mean(x) / max(allMean, eps);
        change = mean(abs(diff(x))) / max(allMean, eps);
        peak = max(x) / max(allMean, eps);
        quiet = mean(a < .6*mean(a));
        ramp = (mean(x(end-4:end))-mean(x(1:5))) / max(allMean, eps);
        dyn(i, (j-1)*6+(1:6)) = single([ratio, change, peak, quiet, fast, ramp]);
    end
end
end
