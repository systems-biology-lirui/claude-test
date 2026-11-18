function res = mem_scan(cal, qry, cfg)
% Compare windows and regularization using the same held-out trials.
[x, names, nBase] = views(cal, cfg);
q = views(qry, cfg);
fold = mem_folds(cal, cfg);
g = [];
for a = 1:nBase
    for lam = cfg.lambda
        for rbf = cfg.rbf
            for rel = cfg.relevance
                p = struct('view', a, 'lam', lam, 'rbf', rbf, 'rel', rel, 'joint', false);
                g = [g, p]; %#ok<AGROW>
            end
        end
    end
end
for a = nBase+1:numel(names)
    for lam = [.12 .70]
        for rel = [0 .75]
            p = struct('view', a, 'lam', lam, 'rbf', 0, 'rel', rel, 'joint', false);
            g = [g, p]; %#ok<AGROW>
        end
    end
end
% Test whether accounting for the odor helps decode behavior.
for a = [1 10 13 15 16 20]
    for lam = cfg.lambda
        p = struct('view', a, 'lam', lam, 'rbf', 0, 'rel', .75, 'joint', true);
        g = [g, p]; %#ok<AGROW>
    end
end
n = numel(cal.id);
nq = numel(qry.id);
cv = zeros(n, 2, cfg.reps, numel(g), 'single');
pred = zeros(nq, 2, numel(g), 'single');
y = cal.y;
parfor (i = 1:numel(g), cfg.workers)
    p = g(i);
    xx = x{p.view};
    qq = q{p.view};
    held = zeros(n, 2, cfg.reps, 'single');
    for rep = 1:cfg.reps
        for f = 1:cfg.folds
            va = fold(:, rep) == f;
            held(va, :, rep) = fit(xx(~va, :), y(~va, :), xx(va, :), p);
        end
    end
    cv(:, :, :, i) = held;
    pred(:, :, i) = fit(xx, y, qq, p);
end
res.cv = cv;
res.pred = pred;
res.grid = g;
res.names = names([g.view]);
res.names([g.joint]) = res.names([g.joint]) + "_per_odor";
end

function s = fit(x, y, q, p)
s = mem_ridge(x, y, q, p);
if ~p.joint || nnz(y(:, 1)) < 5 || nnz(~y(:, 1)) < 5
    return
end
a = y(:, 1);
s0 = mem_ridge(x(~a, :), y(~a, 2), q, p);
s1 = mem_ridge(x(a, :), y(a, 2), q, p);
p1 = max(0, min(1, 1.5 * (s(:, 1) - .5) + .5));
s(:, 2) = .5 * s(:, 2) + .5 * ((1-p1).*s0 + p1.*s1);
end

function [x, names, nBase] = views(d, cfg)
f = sqrt(d.f);
n = numel(d.id);
win = @(cols) reshape(f(:, :, cols), n, []);
names = ["mean", "start_50ms", "start_100ms", "start_250ms", "start_500ms", ...
    "start_1s", "start_1_5s", "start_scales", "start_steps", "mid", ...
    "last_250ms", "last_500ms", "last_1s", "end_steps", "early_late", ...
    "delay_steps", "end_change", "pop_bursts", "raw_early", "raw_mean"];
x = {win(1), win(2), win(3), win(4), win(5), win(6), win(7), ...
    win([2 3 4 5 6]), win([2 8 9 10 11 12]), win(14), ...
    win(17), win(19), win(18), win([15 16 17 20]), win([5 6 1 18 19]), ...
    win([6 22 23 18]), [win(17)-win(1), win(19)-win(1), win(18)-win(1)], ...
    d.pop, reshape(d.f(:, :, [4 5 6]), n, []), reshape(d.f(:, :, 1), n, [])};
nBase = numel(names);
if ~isfield(cfg, 'useExtra') || ~cfg.useExtra
    return
end

names = [names, "population_dynamics", "population_bursts_and_dynamics"];
x = [x, {d.dyn, [d.pop, d.dyn]}];
% Keep each anatomical group intact so its strongest units are not averaged away.
seen = false(size(d.reg, 1), 0);
for lv = [2 3]
    labels = unique(d.reg(:, lv));
    for j = 1:numel(labels)
        use = d.reg(:, lv) == labels(j);
        if labels(j) == "" || nnz(use) < max(12, .02*size(f, 2)) || any(all(seen == use, 1))
            continue
        end
        seen(:, end+1) = use; %#ok<AGROW>
        tag = "region_" + labels(j);
        names = [names, tag + ["_early", "_whole", "_trajectory", "_late"]]; %#ok<AGROW>
        x = [x, {reshape(f(:, use, [3 4 5]), n, []), f(:, use, 1), ...
            reshape(f(:, use, [6 22 23 18]), n, []), reshape(f(:, use, [15 16 17]), n, [])}]; %#ok<AGROW>
    end
end
end
