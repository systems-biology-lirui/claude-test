% Read back the exported ZIP and match each prediction ID to its official NWB.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
team = '谁说bio不如com';
zipPath = fullfile(root, 'output', 'memory', [team '.zip']);
if exist('memoryZipToCheck', 'var')
    zipPath = memoryZipToCheck;
end
[ok, base] = fileattrib(tempdir);
assert(ok);
tmp = tempname(base.Name);
assert(strcmpi(fileparts(tmp), base.Name));
mkdir(tmp);
clean = onCleanup(@() rmdir(tmp, 's'));
unzip(zipPath, fullfile(tmp, 'out'));
unzip(fullfile(root, 'player_data.zip'), fullfile(tmp, 'src'));

files = dir(fullfile(tmp, 'out'));
files = files(~ismember({files.name}, {'.', '..'}));
assert(numel(files) == 4 && ~any([files.isdir]) && all(endsWith({files.name}, '.h5')));
src = dir(fullfile(tmp, 'src', 'player_data', 'challenge*', '*-eval*.nwb'));
src = src(endsWith({src.name}, {'-eval.nwb', '-eval-2.nwb'}));
assert(numel(src) == 4);
keys = {'trial_indices', 'mem_pred_lbl', 'corr_pred_lbl'};
n = 0;
for j = 1:numel(src)
    stem = regexprep(src(j).name, '-eval(-2)?\.nwb$', '');
    name = [team '_' stem '.h5'];
    assert(any(strcmp({files.name}, name)), 'Missing output %s.', name);
    ascii = fullfile(tmp, sprintf('verify%d.h5', j));
    [ok, msg] = copyfile(fullfile(tmp, 'out', name), ascii);
    assert(ok, '%s', msg);
    meta = h5info(ascii);
    assert(isequal(sort({meta.Datasets.Name}), sort(keys)));
    id = int64(h5read(fullfile(src(j).folder, src(j).name), '/intervals/trials/id'));
    for k = 1:numel(keys)
        key = ['/' keys{k}];
        v = h5read(ascii, key);
        ds = h5info(ascii, key);
        assert(isa(v, 'int64') && isequal(ds.Dataspace.Size, numel(id)));
        if k == 1
            assert(isequal(v, sort(id)), 'Incorrect trial IDs in %s.', name);
        elseif k == 2
            assert(all(ismember(v, [4 8])), 'Incorrect odor labels in %s.', name);
        else
            assert(all(ismember(v, [0 1])), 'Incorrect behavior labels in %s.', name);
        end
    end
    n = n + numel(id);
    fprintf('Verified %s (%d trials)\n', name, numel(id));
end
assert(n == 983);
clear clean
fprintf('Memory ZIP verification passed for all %d evaluation trials.\n', n);
