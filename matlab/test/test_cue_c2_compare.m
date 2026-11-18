% 在与 v2 相同的外层试次划分上比较各项 C2 气味候选。
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab', 'memory'));
cases = mem_cases(mem_read(fullfile(root, 'player_data.zip')));
c = cases([cases.weight] == .30);
cfg.reps = 3;
cfg.folds = 5;
y = c.cal.y(:, 1);
for seed = [42 927 0]
    cfg.seed = seed;
    if seed > 0
        fold = mem_folds(c.cal, cfg);
    else
        % 连续试次分块，每次留出一段，而不是随机打散。
        cfg.seed = 927;
        [~, ix] = sort(c.cal.id);
        fold = zeros(numel(y), 1);
        fold(ix) = floor((0:numel(y)-1) * cfg.folds / numel(y)) + 1;
    end
    pred = [];
    reps = size(fold, 2);
    base = false(numel(y), reps);
    t = tic;
    for rep = 1:reps
        for k = 1:cfg.folds
            va = fold(:, rep) == k;
            d = c;
            d.cal = rows(c.cal, ~va);
            d.qry = rows(c.cal, va);
            d.qry.y = [];
            [s, names] = mem_cue_trials(d, cfg);
            [s, names] = addMix(s, names);
            if isempty(pred)
                pred = false(numel(y), reps, numel(names));
            end
            pred(va, rep, :) = s >= .5;
            p = struct('lam', .10, 'rbf', 0, 'rel', 0);
            base(va, rep) = mem_ridge(d.cal.f(:, :, 1), d.cal.y(:, 1), d.qry.f(:, :, 1), p) >= .5;
        end
    end
    acc = reshape(mean(pred == y, [1 2]), [], 1);
    bal = reshape(.5 * (mean(pred(y, :, :), [1 2]) + mean(~pred(~y, :, :), [1 2])), [], 1);
    fprintf('Seed %d\n', seed);
    disp(table(names.', acc, bal, 'VariableNames', {'Candidate', 'Accuracy', 'Balanced'}));
    fprintf('Raw mean baseline %.4f. Compared %d folds in %.2f seconds.\n', ...
        mean(base == y, 'all'), reps*cfg.folds, toc(t));
end

% 把每个 C3 训练场次当作独立目标，模拟 32 个标注后预测其余试次。
c3 = cases(find([cases.weight] == .15, 1));
src = c3.src;
cfg.seed = 927;
rngSrc = RandStream('mt19937ar', 'Seed', cfg.seed);
t = tic;
for j = 1:numel(src)
    a = src(j);
    tr = false(size(a.id));
    for odor = 0:1
        ix = find(a.y(:, 1) == odor);
        tr(ix(randperm(rngSrc, numel(ix), 16))) = true;
    end
    d.cal = rows(a, tr);
    d.qry = rows(a, ~tr);
    y = d.qry.y(:, 1);
    d.qry.y = [];
    d.src = src((1:numel(src)) ~= j);
    [s, names] = mem_cue_trials(d, cfg);
    [s, names] = addMix(s, names);
    acc = mean((s >= .5) == y, 1).';
    bal = .5 * (mean(s(y, :) >= .5, 1) + mean(s(~y, :) < .5, 1)).';
    fprintf('C3 pseudo-target: %s (%d train, %d held out)\n', a.name, nnz(tr), nnz(~tr));
    disp(table(names.', acc, bal, 'VariableNames', {'Candidate', 'Accuracy', 'Balanced'}));
end
fprintf('Four pseudo-targets in %.2f seconds.\n', toc(t));

function [s, names] = addMix(s, names)
use = names == "early_unit_ridge" | names == "contact_early_centroid_src0.25";
s(:, end+1) = mean(max(0, min(1, s(:, use))), 2);
names(end+1) = "early_ridge_and_contact_src0.25";
end

function d = rows(d, use)
d.id = d.id(use);
d.delay = d.delay(use);
d.f = d.f(use, :, :);
d.y = d.y(use, :);
end
