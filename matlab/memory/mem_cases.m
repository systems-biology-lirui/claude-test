function cases = mem_cases(d)
% Match each final recording with its own labeled data and training sessions.
names = {d.name};
cases = struct('cal', {}, 'src', {}, 'qry', {}, 'weight', {});
for k = find(endsWith(names, {'-eval.nwb', '-eval-2.nwb'}))
    q = d(k);
    if q.ch == 1
        calName = strrep(q.name, '-eval.nwb', '-train.nwb');
        src = [];
        weight = .40;
    else
        calName = strrep(q.name, '-eval-2.nwb', '-eval-1.nwb');
        src = d([d.ch] == q.ch & endsWith(names, '-train.nwb'));
        weight = .15;
        if q.ch == 2
            weight = .30;
        end
    end
    a = d(strcmp(names, calName));
    assert(isscalar(a) && isequal(a.units, q.units) && isequal(a.reg, q.reg), ...
        'Calibration and evaluation must use the same ordered units.');
    assert(~isempty(a.y) && isempty(q.y));
    cases(end+1) = struct('cal', a, 'src', src, 'qry', q, 'weight', weight); %#ok<AGROW>
end
assert(numel(cases) == 4 && abs(sum([cases.weight])-1) < 1e-9);
end
