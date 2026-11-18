% Compare fixed behavior candidates on labeled calibration trials only.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab', 'memory'));
cases = mem_cases(mem_read(fullfile(root, 'player_data.zip')));
cfg.folds = 5;
for j = 1:numel(cases)
    c = cases(j);
    d = c.cal;
    n = numel(d.id);
    y = d.y(:, 2);
    [~, order] = sort(d.id);
    block = zeros(n, 1);
    block(order) = 1 + floor(cfg.folds*(0:n-1)/n);
    cfg.seed = 42;
    cfg.reps = 3; % Match the recorded v2 outer folds.
    v2Fold = mem_folds(d, cfg);
    cfg.seed = 211;
    cfg.reps = 1;
    fold = [block, v2Fold, mem_folds(d, cfg)];
    for k = 1:size(fold, 2)
        cfg.seed = 42; % Keep cue evidence fixed while comparing outer splits.
        for f = 1:cfg.folds
            held = fold(:, k) == f;
            c.cal = take(d, ~held);
            c.qry = take(d, held);
            c.qry.y = [];
            [s, names] = mem_beh_trials(c, cfg);
            if f == 1
                pred = false(n, numel(names));
            end
            pred(held, :) = s >= .5;
        end
        acc(:, k) = mean(pred == y, 1).'; %#ok<SAGROW>
        bal(:, k) = .5*(mean(pred(y, :), 1) + mean(~pred(~y, :), 1)).'; %#ok<SAGROW>
    end
    fprintf('\n%s  (n = %d, correct = %d)\n', d.name, n, nnz(y));
    disp(table(names(:), acc(:, 1), bal(:, 1), acc(:, 2), bal(:, 2), acc(:, 5), bal(:, 5), ...
        mean(acc(:, 2:4), 2), mean(bal(:, 2:4), 2), 'VariableNames', ...
        {'Candidate','BlockAcc','BlockBal','Seed42Acc','Seed42Bal','Seed211Acc','Seed211Bal','V2FoldAcc','V2FoldBal'}));
end

function q = take(d, rows)
q = d;
q.id = d.id(rows);
q.delay = d.delay(rows);
q.y = d.y(rows, :);
q.f = d.f(rows, :, :);
q.pop = d.pop(rows, :);
q.dyn = d.dyn(rows, :);
end
