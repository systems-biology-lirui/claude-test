function res = mem_transfer(c, cfg)
% Fit neural population summaries, using anatomy to align different sessions.
% Every target fold estimates scaling from its labeled training trials only.
cal = c.cal;
qry = c.qry;
src = c.src;
fold = mem_folds(cal, cfg);
cv = zeros(numel(cal.id), 2, cfg.reps, 0, 'single');
pred = zeros(numel(qry.id), 2, 0, 'single');
names = strings(1, 0);
isSrc = false(1, 0);
levels = {[], 1, [2 3]}; % Whole population, broad regions, smaller regions.
for v = 1:numel(levels)
    reg = sharedRegions(cal, src, levels{v});
    x = popFeat(cal, reg);
    q = popFeat(qry, reg);
    sx = [];
    sy = [];
    for j = 1:numel(src)
        a = popFeat(src(j), reg);
        sx = [sx; normFrom(a, a)]; %#ok<AGROW>
        sy = [sy; src(j).y]; %#ok<AGROW>
    end
    mix = 0;
    if ~isempty(src)
        mix = [0 cfg.sourceMix];
    end
    [g1, g2] = ndgrid(cfg.popLambda, mix);
    ls = g1(:).';
    ws = g2(:).';
    count = numel(ls);
    sc = zeros(size(x, 1), 2, cfg.reps, count, 'single');
    pq = zeros(size(q, 1), 2, count, 'single');
    parfor (i = 1:count, cfg.workers)
        lam = ls(i);
        wt = ws(i);
        hold = zeros(size(x, 1), 2, cfg.reps, 'single');
        for r = 1:cfg.reps
            for f = 1:cfg.folds
                va = fold(:, r) == f;
                a = x(~va, :);
                hold(va, :, r) = pooled(normFrom(a, a), cal.y(~va, :), ...
                    normFrom(x(va, :), a), sx, sy, lam, wt);
            end
        end
        sc(:, :, :, i) = hold;
        pq(:, :, i) = pooled(normFrom(x, x), cal.y, normFrom(q, x), sx, sy, lam, wt);
    end
    cv = cat(4, cv, sc);
    pred = cat(3, pred, pq);
    tag = ["population", "broad_regions", "fine_regions"];
    names = [names, tag(v) + "_src" + string(ws)]; %#ok<AGROW>
    isSrc = [isSrc, ws > 0]; %#ok<AGROW>
end
if ~isempty(src) && cfg.useProjection
    [sc, pq, ws] = projectionScores(c, fold, cfg);
    cv = cat(4, cv, sc);
    pred = cat(3, pred, pq);
    names = [names, "projection_src" + string(ws)];
    isSrc = [isSrc, true(1, numel(ws))];
end
res.cv = cv;
res.pred = pred;
res.names = names;
res.isSrc = isSrc;
end

function [cv, pred, ws] = projectionScores(c, fold, cfg)
% Match class evidence across sessions, even when their recorded units differ.
sx = [];
sy = [];
for j = 1:numel(c.src)
    d = c.src(j);
    a = project(d.f, d.y, d.f(1, :, :));
    sx = [sx; normFrom(a, a)]; %#ok<AGROW>
    sy = [sy; d.y]; %#ok<AGROW>
end
x = c.cal.f;
y = c.cal.y;
q = c.qry.f;
[g1, g2] = ndgrid(cfg.popLambda, cfg.sourceMix);
ls = g1(:).';
ws = g2(:).';
m = numel(ls);
cv = zeros(size(y, 1), 2, cfg.reps, m, 'single');
pred = zeros(size(q, 1), 2, m, 'single');
for r = 1:cfg.reps
    for k = 1:cfg.folds
        va = fold(:, r) == k;
        [a, b] = project(x(~va, :, :), y(~va, :), x(va, :, :));
        u = normFrom(a, a);
        v = normFrom(b, a);
        for j = 1:m
            cv(va, :, r, j) = pooled(u, y(~va, :), v, sx, sy, ls(j), ws(j));
        end
    end
end
[a, b] = project(x, y, q);
u = normFrom(a, a);
v = normFrom(b, a);
for j = 1:m
    pred(:, :, j) = pooled(u, y, v, sx, sy, ls(j), ws(j));
end
end

function [a, b] = project(x, y, q)
% First-stage features are leave-one-out scores on the training trials.
win = [3 5 6 1 14 17 18];
n = size(x, 1);
nq = size(q, 1);
a = zeros(n, 3*numel(win));
b = zeros(nq, 3*numel(win));
for j = 1:numel(win)
    z = sqrt(x(:, :, win(j)));
    u = normFrom(z, z);
    v = normFrom(sqrt(q(:, :, win(j))), z);
    scale = max(mean(sum(u.^2, 2)), 1e-6);
    invK = (u*u.'/scale + .2*eye(n)) \ eye(n);
    w = sum(invK, 2);
    bias = (w.' * double(y)) / sum(w);
    coef = invK * (y-bias);
    diagLoo = diag(invK) - w.^2/sum(w);
    sv = y - coef ./ diagLoo;
    sq = v*u.'/scale * coef + bias;
    ids = (j-1)*3+(1:3);
    a(:, ids) = [sv, abs(sv(:, 1)-.5)];
    b(:, ids) = [sq, abs(sq(:, 1)-.5)];
end
end

function reg = sharedRegions(cal, src, levels)
reg = strings(0, 2);
for v = levels
    [labels, ~, g] = unique(cal.reg(:, v));
    for j = find(labels ~= "" & accumarray(g, 1) >= 5).'
        label = labels(j);
        has = arrayfun(@(d) nnz(d.reg(:, v) == label) >= 5, src);
        if isempty(src) || any(has)
            reg(end+1, :) = [string(v), label]; %#ok<AGROW>
        end
    end
end
end

function x = popFeat(d, reg)
% Explainable regional features for early, steady and late neural activity.
f = d.f(:, :, [1 3 5 6 14 17 18]);
n = size(f, 1);
x = [d.pop, summary(f)];
for j = 1:size(reg, 1)
    lv = str2double(reg(j, 1));
    use = d.reg(:, lv) == reg(j, 2);
    if nnz(use) >= 5
        x = [x, summary(f(:, use, :))]; %#ok<AGROW>
    else
        x = [x, zeros(n, 3 * size(f, 3), 'single')]; %#ok<AGROW>
    end
end
end

function x = summary(f)
n = size(f, 1);
s = sqrt(f);
x = [reshape(mean(s, 2), n, []), reshape(std(s, 0, 2), n, []), ...
    reshape(mean(f > 0, 2), n, [])];
x = single(x);
end

function z = normFrom(x, ref)
mu = mean(ref, 1);
sd = std(ref, 0, 1);
use = sd > 1e-6;
z = zeros(size(x));
z(:, use) = double((x(:, use) - mu(use)) ./ sd(use));
z = max(-6, min(6, z));
end

function s = pooled(x, y, q, sx, sy, lam, wt)
% wt is the total weight of source trials relative to target trials.
w = ones(size(x, 1), 1);
if wt > 0
    w = [w; repmat(wt * size(x, 1) / size(sx, 1), size(sx, 1), 1)];
    x = [x; sx];
    y = [y; sy];
end
w = w / sum(w);
y = double(y);
mu = w.' * x;
bias = w.' * y;
a = (x - mu) .* sqrt(w);
b = (y - bias) .* sqrt(w);
if size(a, 2) > size(a, 1)
    coef = a.' * ((a * a.' + lam * eye(size(a, 1))) \ b);
else
    coef = (a.' * a + lam * eye(size(a, 2))) \ (a.' * b);
end
s = (q - mu) * coef + bias;
end
