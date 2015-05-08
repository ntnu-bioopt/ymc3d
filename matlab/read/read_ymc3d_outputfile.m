function [data, dx, dy, dz] = read_ymc3d_outputfile(filename)
	[data, dx, dy, dz] = read_general_mcfile(filename, 'double');
end
