function fold = mem_folds(d, cfg)
% Split complete trials, balancing delay, odor and outcome per fold.
rng(cfg.seed, 'twister');
fold = zeros(numel(d.id), cfg.reps);
[~, ~, group] = unique([d.delay, double(d.y)], 'rows');
for j = 1:cfg.reps
    offset = 0;
    for k = 1:max(group)
        ids = find(group == k);
        ids = ids(randperm(numel(ids)));
        fold(ids, j) = mod(offset + (0:numel(ids)-1), cfg.folds) + 1;
        offset = offset + numel(ids);
    end
end
end
