% 只用官方数据的已标注试次，检查气味候选和输入隔离。
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab', 'memory'));
c = mem_cases(mem_read(fullfile(root, 'player_data.zip')));
c = c([c.weight] == .30);
cfg.seed = 42;
cfg.reps = 1;
cfg.folds = 5;
cfg.cuePanels = 8;
fold = mem_folds(c.cal, cfg);
va = fold == 1;
y = c.cal.y(va, 1);
d = c;
d.cal = rows(c.cal, ~va);
d.qry = rows(c.cal, va);
d.qry.y = [];
[p, names] = mem_cue_trials(d, cfg);
assert(isequal(size(p), [nnz(va), 13]) && all(isfinite(p), 'all'));
assert(numel(unique(names)) == size(p, 2));
assert(isequal(p, mem_cue_trials(d, cfg)));

% 查询行之间互不影响，查询结果标签和两个场次的行为标签不被读取。
b = d;
b.qry.f(1, :, :) = 0;
b.qry.y = [~y, y];
b.cal.y(:, 2) = ~b.cal.y(:, 2);
b.src.y(:, 2) = ~b.src.y(:, 2);
q = mem_cue_trials(b, cfg);
assert(isequal(p(2:end, :), q(2:end, :)));

% 翻转源气味标签只能影响源权重非零的列。
b = d;
b.src.y(:, 1) = ~b.src.y(:, 1);
q = mem_cue_trials(b, cfg);
noSrc = endsWith(names, "src0") | names == "early_unit_ridge";
assert(isequal(p(:, noSrc), q(:, noSrc)));
assert(any(abs(p(:, ~noSrc) - q(:, ~noSrc)) > 1e-8, 'all'));
b = d;
b.src = [];
[q, tags] = mem_cue_trials(b, cfg);
assert(isequal(names, tags) && all(isfinite(q), 'all'));
assert(isequal(q(:, 2:7), q(:, [2 2 2 5 5 5])));

acc = mean((p >= .5) == y, 1);
disp(table(names.', acc.', 'VariableNames', {'Candidate', 'Accuracy'}));

function d = rows(d, use)
d.id = d.id(use);
d.delay = d.delay(use);
d.f = d.f(use, :, :);
d.y = d.y(use, :);
d.pop = d.pop(use, :);
d.dyn = d.dyn(use, :);
end
