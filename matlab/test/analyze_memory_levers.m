% Check the per-fold behavior baseline and whether contact IDs span days.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab', 'memory'));
c = mem_cases(mem_read(fullfile(root, 'player_data.zip')));
cfg.folds = 5;
cfg.reps = 3;
cfg.seed = 42;
weighted = 0;
for j = 1:numel(c)
    d = c(j).cal;
    y = d.y(:, 2);
    fold = mem_folds(d, cfg);
    pred = false(numel(y), cfg.reps);
    for rep = 1:cfg.reps
        for k = 1:cfg.folds
            va = fold(:, rep) == k;
            pred(va, rep) = mean(y(~va)) >= .5;
        end
    end
    acc = mean(pred == y, 'all');
    weighted = weighted + c(j).weight * acc;
    fprintf('Majority %s: %d/%d correct labels, held-out accuracy %.4f\n', ...
        d.name, nnz(y), numel(y), acc);

    subject = extractBefore(d.name, '_');
    for s = c(j).src
        if ~startsWith(s.name, [char(subject) '_'])
            continue
        end
        source = s.units(:, 3:4); % shank and channel, not cluster ID
        target = d.units(:, 3:4);
        shared = intersect(source, target, 'rows');
        annotated = 0;
        agree = 0;
        for q = 1:size(shared, 1)
            a = unique(s.reg(all(source == shared(q, :), 2), 3));
            b = unique(d.reg(all(target == shared(q, :), 2), 3));
            if any(a ~= "") && any(b ~= "")
                annotated = annotated + 1;
                agree = agree + ~isempty(intersect(a(a ~= ""), b(b ~= "")));
            end
        end
        fprintf('Same mouse %s vs %s: %d / %d distinct contacts, %d shared, %d/%d region overlap\n', ...
            s.name, d.name, size(unique(source, 'rows'), 1), size(unique(target, 'rows'), 1), ...
            size(shared, 1), agree, annotated);
    end
end
fprintf('Weighted majority behavior accuracy: %.4f\n', weighted);
