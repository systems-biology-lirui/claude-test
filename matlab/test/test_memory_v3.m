% Run the common paired comparison and verify any selected new package.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab'));
out = run_memory('v3');
assert(abs(out.base - .7449) < .0001, 'The published v2 validation did not reproduce.');
assert(isfile(out.report));
if out.pick > 1
    assert(isfile(out.zip) && all(out.acc(1:2, out.pick) > out.acc(1:2, 1)));
    memoryZipToCheck = out.zip;
    run(fullfile(root, 'matlab', 'test', 'verify_memory_zip.m'));
    clear memoryZipToCheck
end
