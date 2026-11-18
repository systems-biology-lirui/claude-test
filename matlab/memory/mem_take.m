function d = mem_take(d, rows)
% Select complete trials without changing the recording's unit order.
d.id = d.id(rows);
d.delay = d.delay(rows);
if ~isempty(d.y)
    d.y = d.y(rows, :);
end
d.f = d.f(rows, :, :);
d.pop = d.pop(rows, :);
d.dyn = d.dyn(rows, :);
end
