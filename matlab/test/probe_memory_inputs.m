% Inspect the allowed memory input fields and the example submission.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
[ok, base] = fileattrib(tempdir);
assert(ok);
tmp = tempname(base.Name);
assert(strcmpi(fileparts(tmp), base.Name));
mkdir(tmp);
clean = onCleanup(@() rmdir(tmp, 's'));
unzip(fullfile(root, 'player_data.zip'), tmp);

fprintf('MATLAB %s, max BLAS threads %d, parallel license %d, statistics license %d\n', ...
    version, maxNumCompThreads, license('test', 'Distrib_Computing_Toolbox'), ...
    license('test', 'Statistics_Toolbox'));
fprintf('fitcsvm: %s\n', which('fitcsvm'));
fprintf('parpool: %s\n', which('parpool'));

data = fullfile(tmp, 'player_data');
files = dir(fullfile(data, 'challenge*', '*.nwb'));
fields = '/intervals/trials/';
for k = 1:numel(files)
    fn = fullfile(files(k).folder, files(k).name);
    if endsWith(files(k).name, {'-train.nwb','-eval-1.nwb'})
        ids = h5read(fn, [fields 'id']);
        y1 = h5read(fn, [fields 'sample_cue']);
        y2 = h5read(fn, [fields 'is_correct']);
        if iscell(y2) || isstring(y2)
            y2 = strcmpi(y2, 'TRUE');
        end
        reg = h5read(fn, '/units/anatomic_level_6');
        fprintf('%s: %d trials, cue8 %d, correct %d, %d regions\n', ...
            files(k).name, numel(ids), nnz(y1 == 8), nnz(y2), numel(unique(string(reg))));
    end
end

files = dir(fullfile(data, 'challenge1', '*-train.nwb'));
fn = fullfile(files(1).folder, files(1).name);
u = h5info(fn, '/units');
disp({u.Datasets.Name});
sp = h5read(fn, '/units/spike_times');
ix = h5read(fn, '/units/spike_times_index');
fprintf('Spike matrix %s %s, ragged index %s %s, last index %d\n', ...
    mat2str(size(sp)), class(sp), mat2str(size(ix)), class(ix), ix(end));
disp(sp(:, 1:12));
disp(sp(:, end-6:end));
for level = 3:8
    reg = string(h5read(fn, sprintf('/units/anatomic_level_%d', level)));
    fprintf('Level %d: %s\n', level, strjoin(unique(reg), ', '));
end

example = dir(fullfile(data, 'challenge1', 'example*.h5'));
fn = fullfile(example.folder, example.name);
s = h5info(fn);
for k = 1:numel(s.Datasets)
    field = s.Datasets(k);
    values = h5read(fn, ['/' field.Name]);
    fprintf('Example %s: file dimensions %s, %s, %d unique, range [%d, %d]\n', ...
        field.Name, mat2str(field.Dataspace.Size), class(values), numel(unique(values)), ...
        min(values), max(values));
end
clear clean
