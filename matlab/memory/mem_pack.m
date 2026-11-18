function zipPath = mem_pack(pred, cfg)
% Write the three 1D int64 arrays requested by the official example.
if ~isfolder(cfg.out)
    mkdir(cfg.out);
end
[ok, base] = fileattrib(cfg.out);
assert(ok);
tmp = tempname(base.Name);
assert(strcmpi(fileparts(tmp), base.Name), 'Unexpected ZIP staging path.');
mkdir(tmp);
clean = onCleanup(@() rmdir(tmp, 's')); %#ok<NASGU>
assert(numel(pred) == 4, 'Expected four evaluation files.');
names = cell(1, numel(pred));
keys = {'trial_indices', 'mem_pred_lbl', 'corr_pred_lbl'};
for j = 1:numel(pred)
    a = pred(j);
    [id, order] = sort(a.id);
    assert(size(a.y, 1) == numel(id) && size(a.y, 2) == 2);
    values = int64([id(:), 4 + 4*double(a.y(order, 1)), double(a.y(order, 2))]);
    draft = fullfile(tmp, sprintf('file%d.h5', j));
    for k = 1:numel(keys)
        key = ['/' keys{k}];
        h5create(draft, key, numel(id), 'Datatype', 'int64');
        h5write(draft, key, values(:, k));
        ds = h5info(draft, key);
        assert(isscalar(ds.Dataspace.Size) && isequal(h5read(draft, key), values(:, k)));
    end
    stem = regexprep(a.name, '-eval(-2)?\.nwb$', '');
    names{j} = [cfg.team '_' stem '.h5'];
    [ok, msg] = movefile(draft, fullfile(tmp, names{j}));
    assert(ok, '%s', msg);
end
draftZip = fullfile(tmp, 'submission.zip');
zip(draftZip, names, tmp);

% Check the ZIP directory before promoting the completed file.
zf = javaObject('java.util.zip.ZipFile', draftZip);
zclose = onCleanup(@() zf.close());
assert(zf.size() == numel(names), 'Wrong number of ZIP entries.');
for j = 1:numel(names)
    ent = zf.getEntry(names{j});
    assert(~isempty(ent) && ent.getSize() > 0, 'Missing ZIP entry: %s', names{j});
end
clear zclose
zipPath = fullfile(cfg.out, [cfg.team '.zip']);
[ok, msg] = movefile(draftZip, zipPath, 'f');
assert(ok, '%s', msg);
end
