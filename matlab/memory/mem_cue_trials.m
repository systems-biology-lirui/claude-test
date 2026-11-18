function [score, names] = mem_cue_trials(c, cfg)
% 只预测气味。所有中心、方向和源场次权重仅用 cal 或 src 拟合。
a = c.cal;
q = c.qry;
y = a.y(:, 1);
early = [2 3 4 5 6];
span = [3 5 6 1 14 17 18];
mix = [0 .25 .50];
panels = 8;
if isfield(cfg, 'cuePanels')
    panels = cfg.cuePanels;
end
x = reshape(sqrt(a.f(:, :, early)), numel(a.id), []);
v = reshape(sqrt(q.f(:, :, early)), numel(q.id), []);
p = struct('lam', .20, 'rbf', 0, 'rel', .75);
score = mem_ridge(x, y, v, p);
names = "early_unit_ridge";
wins = {early, span};
tag = ["early", "span"];
for kind = 1:2
    win = wins{kind};
    rngSrc = RandStream('mt19937ar', 'Seed', cfg.seed);
    [x, v] = project(a, q.f, win, rngSrc);
    [x, v] = scale(x, v);
    sx = zeros(0, numel(win));
    sy = false(0, 1);
    for i = 1:numel(c.src)
        d = c.src(i);
        if size(d.y, 1) < numel(y)
            continue
        end
        for j = 1:panels
            ids = panel(a, d, rngSrc);
            p = struct('f', d.f(ids, :, :), 'y', d.y(ids, 1), 'delay', d.delay(ids));
            z = project(p, d.f([], :, :), win, rngSrc);
            sx = [sx; scale(z, z)]; %#ok<AGROW>
            sy = [sy; p.y]; %#ok<AGROW>
        end
    end
    for w = mix
        score(:, end+1) = pooled(x, y, v, sx, sy, w); %#ok<AGROW>
        names(end+1) = tag(kind) + "_matched_src" + string(w); %#ok<AGROW>
    end
end

% 分组平均放电率，再按各场次的两种气味中心对齐。
for kind = ["region", "contact"]
    [grp, src] = groups(a, c.src, kind);
    if isempty(grp)
        grp = {1:size(a.f, 2)};
        src = cell(1, numel(c.src));
    end
    x = groupRates(a, grp, early);
    v = groupRates(q, grp, early);
    [x, v] = scale(x, v);
    [dir, mid] = centers(x, y);
    sd = zeros(size(dir));
    n = zeros(size(dir));
    for j = 1:numel(c.src)
        d = c.src(j);
        u = groupRates(d, src(:, j), early);
        u = scale(u, u);
        use = std(u, 0, 1) > 0 & any(d.y(:, 1)) & any(~d.y(:, 1));
        ds = centers(u, d.y(:, 1));
        sd(use) = sd(use) + ds(use);
        n(use) = n(use) + 1;
    end
    sd = sd ./ max(n, 1);
    for w = mix
        b = (dir + w*sd) ./ (1 + w*(n > 0));
        s = (v - mid) * b.' / max(sum(b.^2), eps);
        score(:, end+1) = .5 + s; %#ok<AGROW>
        names(end+1) = kind + "_early_centroid_src" + string(w); %#ok<AGROW>
    end
end
end

function ids = panel(a, d, rngSrc)
% 按气味和延迟分层，源场次样本数等于本次目标校准样本数。
left = true(size(d.y, 1), 1);
ids = zeros(0, 1);
want = zeros(2, 1);
for odor = 0:1
    y = a.y(:, 1) == odor;
    for delay = unique(a.delay).'
        count = nnz(y & a.delay == delay);
        ix = find(left & d.y(:, 1) == odor & d.delay == delay);
        ix = ix(randperm(rngSrc, numel(ix), min(count, numel(ix))));
        left(ix) = false;
        ids = [ids; ix]; %#ok<AGROW>
        want(odor+1) = want(odor+1) + count - numel(ix);
    end
end
for odor = 0:1
    ix = find(left & d.y(:, 1) == odor);
    ix = ix(randperm(rngSrc, numel(ix), min(want(odor+1), numel(ix))));
    left(ix) = false;
    ids = [ids; ix]; %#ok<AGROW>
end
ix = find(left);
ix = ix(randperm(rngSrc, numel(ix), numel(a.id) - numel(ids)));
ids = [ids; ix];
end

function [a, b] = project(d, q, win, rngSrc)
% 每个窗口用五折预测。折内标准化与方向拟合都不含留出的试次。
y = d.y(:, 1);
n = numel(y);
a = zeros(n, numel(win));
b = zeros(size(q, 1), numel(win));
fold = zeros(n, 1);
nf = min(5, n);
[~, ~, group] = unique([double(y), d.delay], 'rows');
offset = 0;
for k = 1:max(group)
    ids = find(group == k);
    ids = ids(randperm(rngSrc, numel(ids)));
    fold(ids) = mod(offset + (0:numel(ids)-1), nf) + 1;
    offset = offset + numel(ids);
end
p = struct('lam', .20, 'rbf', 0, 'rel', 0);
for j = 1:numel(win)
    x = sqrt(d.f(:, :, win(j)));
    for k = 1:nf
        va = fold == k;
        if all(va)
            a(va, j) = .5;
        else
            yt = y(~va);
            a(va, j) = mem_ridge(x(~va, :), yt, x(va, :), p) + .5 - mean(yt);
        end
    end
    if ~isempty(q)
        b(:, j) = mem_ridge(x, y, sqrt(q(:, :, win(j))), p) + .5 - mean(y);
    end
end
end

function [x, q] = scale(x, q)
x = double(x);
q = double(q);
mu = mean(x, 1);
sd = std(x, 0, 1);
use = sd > 1e-6;
x(:, ~use) = 0;
q(:, ~use) = 0;
if any(use)
    s = max(sd(use), .30 * median(sd(use)));
    x(:, use) = max(-6, min(6, (x(:, use) - mu(use)) ./ s));
    q(:, use) = max(-6, min(6, (q(:, use) - mu(use)) ./ s));
end
end

function s = pooled(x, y, q, sx, sy, wt)
w = ones(size(x, 1), 1);
if wt > 0 && ~isempty(sy)
    w = [w; repmat(wt*numel(y)/numel(sy), numel(sy), 1)];
    x = [x; sx];
    y = [y; sy];
end
w = w / sum(w);
mu = w.' * x;
bias = w.' * double(y);
a = (x - mu) .* sqrt(w);
b = (double(y) - bias) .* sqrt(w);
s = (q - mu) * ((a.'*a + .20*eye(size(a, 2))) \ (a.'*b)) + bias;
end

function [grp, src] = groups(a, sessions, kind)
grp = cell(0, 1);
src = cell(0, numel(sessions));
if kind == "region"
    % 细脑区各自匹配，至少有五个单元。
    labels = unique(a.reg(:, 3));
    for label = labels(labels ~= "").'
        use = a.reg(:, 3) == label;
        if nnz(use) < 5
            continue
        end
        ss = cell(1, numel(sessions));
        for j = 1:numel(sessions)
            u = sessions(j).reg(:, 3) == label;
            if nnz(u) >= 5
                ss{j} = find(u);
            end
        end
        if any(~cellfun(@isempty, ss))
            grp{end+1, 1} = find(use); %#ok<AGROW>
            src(end+1, :) = ss; %#ok<AGROW>
        end
    end
    return
end

% 同一受试个体、同一电极触点、同一细脑区才归为一组。
subject = extractBefore(string(a.name), "_");
loc = a.units(:, 3:4);
for row = unique(loc, 'rows').'
    at = all(loc == row.', 2);
    labels = unique(a.reg(at, 3));
    for label = labels(labels ~= "").'
        use = at & a.reg(:, 3) == label;
        ss = cell(1, numel(sessions));
        for j = 1:numel(sessions)
            d = sessions(j);
            if extractBefore(string(d.name), "_") ~= subject
                continue
            end
            u = all(d.units(:, 3:4) == row.', 2) & d.reg(:, 3) == label;
            ss{j} = find(u);
        end
        if any(~cellfun(@isempty, ss))
            grp{end+1, 1} = find(use); %#ok<AGROW>
            src(end+1, :) = ss; %#ok<AGROW>
        end
    end
end
end

function x = groupRates(d, grp, win)
n = numel(d.id);
x = zeros(n, numel(win)*numel(grp));
for j = 1:numel(grp)
    if ~isempty(grp{j})
        a = mean(sqrt(d.f(:, grp{j}, win)), 2);
        x(:, (j-1)*numel(win)+(1:numel(win))) = reshape(a, n, []);
    end
end
end

function [dir, mid] = centers(x, y)
dir = zeros(1, size(x, 2));
mid = mean(x, 1);
if any(y) && any(~y)
    hi = mean(x(y, :), 1);
    lo = mean(x(~y, :), 1);
    dir = hi - lo;
    mid = .5*(hi + lo);
end
end
