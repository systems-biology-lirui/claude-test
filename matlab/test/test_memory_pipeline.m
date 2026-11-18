% Run the normal pipeline and inspect the final ZIP's four HDF5 files.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab'));
out = run_memory();
assert(numel(out.pred) == 4 && isfinite(out.score));
[~, team] = fileparts(out.zip);
assert(strcmp(team, char([35841 35828 98 105 111 19981 22914 99 111 109])));

[ok, base] = fileattrib(tempdir);
assert(ok);
tmp = tempname(base.Name);
assert(strcmpi(fileparts(tmp), base.Name));
mkdir(tmp);
clean = onCleanup(@() rmdir(tmp, 's'));
unzip(out.zip, tmp);
files = dir(fullfile(tmp, '*.h5'));
entries = dir(tmp);
entries = entries(~ismember({entries.name}, {'.', '..'}));
assert(numel(files) == 4 && numel(entries) == 4 && ~any([entries.isdir]));
fields = {'trial_indices', 'mem_pred_lbl', 'corr_pred_lbl'};
assert(isequal(sort(arrayfun(@(a) numel(a.id), out.pred)), [162 245 288 288]));
for j = 1:numel(out.pred)
    p = out.pred(j);
    stem = regexprep(p.name, '-eval(-2)?\.nwb$', '');
    name = [team '_' stem '.h5'];
    assert(any(strcmp(name, {files.name})), 'Missing output %s.', name);
    fp = fullfile(tmp, name);
    ascii = fullfile(tmp, sprintf('verify%d.h5', j));
    [ok, msg] = copyfile(fp, ascii);
    assert(ok, '%s', msg);
    meta = h5info(ascii);
    assert(isequal(sort({meta.Datasets.Name}), sort(fields)));
    [id, order] = sort(p.id);
    want = int64([id, 4 + 4*double(p.y(order, 1)), double(p.y(order, 2))]);
    for k = 1:numel(fields)
        key = ['/' fields{k}];
        actual = h5read(ascii, key);
        assert(isa(actual, 'int64') && isequal(actual, want(:, k)));
        ds = h5info(ascii, key);
        assert(isequal(ds.Dataspace.Size, numel(id)), 'Expected a rank-1 dataset.');
    end
    fprintf('Verified %s: %d trial IDs and both label arrays.\n', name, numel(id));
end
clear clean
fprintf('Memory submission ZIP verification passed.\n');
