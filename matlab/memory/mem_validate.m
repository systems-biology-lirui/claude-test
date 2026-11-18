function out = mem_validate(c, cfg)
% Outer folds never take part in window selection or score calibration.
outer = cfg;
outer.folds = cfg.outerFolds;
outer.reps = cfg.outerReps;
fold = mem_folds(c.cal, outer);
inner = cfg;
inner.reps = cfg.innerReps;
n = numel(c.cal.id);
pred = false(n, 2, outer.reps);
base = false(n, 2, outer.reps);
for j = 1:outer.reps
    for i = 1:outer.folds
        va = fold(:, j) == i;
        k = c;
        k.cal = mem_take(c.cal, ~va);
        k.qry = mem_take(c.cal, va);
        k.qry.y = [];
        pred(va, :, j) = mem_predict(k, inner);
        p = struct('lam', .10, 'rbf', 0, 'rel', 0);
        base(va, :, j) = mem_ridge(k.cal.f(:, :, 1), k.cal.y, k.qry.f(:, :, 1), p) >= .50;
    end
end
out.acc = mean(pred == c.cal.y, [1 3]);
out.base = mean(base == c.cal.y, [1 3]);
out.bal = zeros(1, 2);
for j = 1:2
    y = c.cal.y(:, j);
    out.bal(j) = (mean(pred(y, j, :), 'all') + mean(~pred(~y, j, :), 'all')) / 2;
end
end
