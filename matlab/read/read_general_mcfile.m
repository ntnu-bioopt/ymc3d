function [data, dx, dy, dz] = read_general_mcfile(filename, datatype)
	fid = fopen(filename);
	
	%read number of dimensions
	num_dims = fread(fid, 1, 'integer*4');

	%geometry sizes
	num_x = fread(fid, 1, 'integer*4');
	num_y = fread(fid, 1, 'integer*4');
	num_z = 1;
	if (num_dims > 2)
		num_z = fread(fid, 1, 'integer*4');
	end

	%grid sizes
	dx = fread(fid, 1, 'double');
	dy = fread(fid, 1, 'double');
	dz = 0;
	if (num_dims > 2)
		dz = fread(fid, 1, 'double');
	end

	%data
	data = zeros(num_x, num_y, num_z);
	for k=1:num_z
		for j=1:num_y
			data(:, j, k) = fread(fid, num_x, datatype);
		end	
	end
end
