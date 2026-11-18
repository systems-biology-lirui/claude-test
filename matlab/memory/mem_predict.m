function [pred, info, exp] = mem_predict(c, cfg)
% Choose and blend models strictly on labeled cross-validation predictions.
a = mem_scan(c.cal, c.qry, cfg);
b = mem_transfer(c, cfg);
cv = cat(4, a.cv, b.cv);
q = cat(3, a.pred, b.pred);
names = [a.names, b.names];
src = [false(1, numel(a.names)), b.isSrc];
joint = [[a.grid.joint], false(1, numel(b.names))];
n = numel(c.cal.id);
nq = numel(c.qry.id);
pred = false(nq, 2);
info.count = numel(names);
for j = 1:2
    y = c.cal.y(:, j);
    s = reshape(double(cv(:, j, :, :)), n, cfg.reps, []);
    sq = reshape(double(q(:, j, :)), nq, []);
    use = ~src;
    if j == 1 && isfield(cfg, 'useExtra') && cfg.useExtra
        % The per-odor model only changes behavior predictions.
        use = use & ~joint;
    end
    [v, p, chosen] = blend(s(:, :, use), sq(:, use), names(use), y, cfg.bestViews, cfg.blendTemp);
    [v, p, wt] = mixSource(v, p, s, sq, names, src, y, cfg);
    cut = .5;
    if j == 2
        [p, strength, v] = calibrateBehavior(v, p, y, cfg);
        info.behaviorStrength = strength;
    end
    pred(:, j) = p >= cut;
    info.names{j} = chosen;
    info.srcWeight(j) = wt;
    if nargout > 2
        exp.score(:, j) = p;
        exp.cv(:, j, :) = reshape(v, n, 1, cfg.reps);
        if j == 1
            use = ~src & ~joint;
            [va, pa] = blend(s(:, :, use), sq(:, use), names(use), y, cfg.bestViews, cfg.blendTemp);
            [vb, pb] = mixSource(va, pa, s, sq, names, src, y, cfg);
            exp.cueScore = [pb, pa];
            exp.cueCV = cat(3, vb, va);
        end
    end
end
end

function [v, p, wt] = mixSource(v, p, s, sq, names, src, y, cfg)
wt = 0;
if ~any(src)
    return
end
[vs, ps] = blend(s(:, :, src), sq(:, src), names(src), y, 2, cfg.blendTemp);
best = -inf;
for w = cfg.srcWeights
    z = (1-w)*v + w*vs;
    val = quality(z, y) - .012*w;
    if val > best
        best = val;
        wt = w;
    end
end
v = (1-wt)*v + wt*vs;
p = (1-wt)*p + wt*ps;
end

function [p, strength, v] = calibrateBehavior(v, p, y, cfg)
% Require at least one extra correct trial before trusting a weak signal.
prior = mean(y);
v = max(0, min(1, v));
p = max(0, min(1, p));
best = max(prior, 1-prior) + cfg.behaviorGain/numel(y);
strength = cfg.behaviorStrength(1);
for w = cfg.behaviorStrength(2:end)
    acc = mean(((prior + w*(v-prior)) >= .50) == y, 'all');
    if acc > best
        best = acc;
        strength = w;
    end
end
p = prior + strength*(p-prior);
v = prior + strength*(v-prior);
end

function [v, p, chosen] = blend(s, q, names, y, count, temp)
% Count each feature view once so a larger grid cannot dominate the blend.
qv = reshape(quality(s, y), 1, []);
[groups, ~, group] = unique(names, 'stable');
val = zeros(1, numel(groups));
cv = zeros(size(s, 1), size(s, 2), numel(groups));
pred = zeros(size(q, 1), numel(groups));
for k = 1:numel(groups)
    ids = find(group == k);
    [~, ix] = sort(qv(ids), 'descend');
    ids = ids(ix(1:min(2, numel(ix))));
    val(k) = mean(qv(ids));
    cv(:, :, k) = mean(s(:, :, ids), 3);
    pred(:, k) = mean(q(:, ids), 2);
end
[vBest, ix] = sort(val, 'descend');
ix = ix(1:min(count, numel(ix)));
w = exp((val(ix)-vBest(1)) / temp);
w = w / sum(w);
v = sum(cv(:, :, ix) .* reshape(w, 1, 1, []), 3);
p = pred(:, ix) * w.';
chosen = groups(ix);
end

function z = quality(s, y)
acc = mean((s >= .50) == y, [1 2]);
brier = mean((max(0, min(1, s)) - y).^2, [1 2]);
z = .80*acc + .20*(1-brier);
end
