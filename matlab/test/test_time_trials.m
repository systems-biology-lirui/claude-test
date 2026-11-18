% Compare odor candidates on repeated folds of four labeled calibration sets.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab', 'memory'));
c = mem_cases(mem_read(fullfile(root, 'player_data.zip')));
cfg.seed = 42;
cfg.folds = 5;
cfg.reps = 3;
cfg.timeLda = .90;
cfg.timeRbf = 1;
p = struct('lam', .20, 'rbf', 0, 'rel', .75);
[~, names] = mem_time_trials(c(1), cfg);
names = [names, "early_ridge", "whole_ridge"];
acc = zeros(numel(c), numel(names));

for i = 1:numel(c)
    a = c(i).cal;
    q = c(i).qry;
    assert(size(a.f, 3) >= 26 && size(q.f, 3) >= 26);
    assert(max(abs(mean(a.f(:, :, 24:26), 3) - a.f(:, :, 1)), [], 'all') < 1e-4);
    assert(max(abs(mean(q.f(:, :, 24:26), 3) - q.f(:, :, 1)), [], 'all') < 1e-4);
    [score, tags] = mem_time_trials(c(i), cfg);
    assert(isequal(tags, names(1:end-2)) && numel(unique(tags)) == numel(tags));
    assert(isequal(size(score), [numel(q.id), numel(tags)]) && all(isfinite(score), 'all'));
    assert(isequal(score, mem_time_trials(c(i), cfg)));

    % A changed query row or a changed behavior label leaves other rows alone.
    d = c(i);
    d.qry.f(1, :, :) = 0;
    d.qry.y = true(numel(q.id), 2);
    d.cal.y(:, 2) = ~d.cal.y(:, 2);
    other = mem_time_trials(d, cfg);
    assert(isequal(score(2:end, :), other(2:end, :)));

    fold = mem_folds(a, cfg);
    y = a.y(:, 1);
    held = false(numel(y), cfg.reps, size(acc, 2));
    early = reshape(sqrt(a.f(:, :, 2:6)), numel(y), []);
    whole = sqrt(a.f(:, :, 1));
    for rep = 1:cfg.reps
        for f = 1:cfg.folds
            va = fold(:, rep) == f;
            d.cal = mem_take(a, ~va);
            d.qry = mem_take(a, va);
            d.qry.y = [];
            held(va, rep, 1:end-2) = mem_time_trials(d, cfg) >= .5;
            held(va, rep, end-1) = mem_ridge(early(~va, :), y(~va), early(va, :), p) >= .5;
            held(va, rep, end) = mem_ridge(whole(~va, :), y(~va), whole(va, :), p) >= .5;
        end
    end
    acc(i, :) = reshape(mean(held == y, [1 2]), 1, []);
    fprintf('\n%s (%d trials)\n', q.name, numel(y));
    disp(table(names.', acc(i, :).', 'VariableNames', {'Candidate', 'OdorAccuracy'}));
end

w = [c.weight];
disp(table(names.', (w*acc).', 'VariableNames', {'Candidate', 'WeightedOdorAccuracy'}));
