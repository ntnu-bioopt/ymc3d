function val = melanin(lambda) %(m-1)
	val = (lambda/694)^(-3.46);
end

function val = musr(lambda) %(m-1)
	aMie = 18.780;
	bMie = 0.22;
	aRay = 17.6;
	val = (100*(aMie*(lambda/500.0)^(-bMie) + aRay*(lambda/500)^(-4)));
end

function val = g(lambda)
	val = 0.62 + lambda*29e-05;
end

function prepare_optprops(lambdas, name)
	%this is a very simple example for readability where I have reduced the amount of used optical properties in order to skip all the optical property files. Not the same example as the one used in the paper. 
	% 1: epidermis
	% 2: papillary dermis
	% 3: reticular dermis


	% load blood values
	load('muab.mat');
	mua_deoxy = @(x) interp1(muabd(:, 1), muabd(:,2), x);
	mua_oxy = @(x) interp1(muabo(:,1), muabo(:,2), x);
	mua_blood = @(oxy, x) mua_oxy*oxy + mua_deoxy*(1-oxy);

	for i = 1:length(lambdas)
		lambda = lambdas(i);
		optical_data = zeros(3, 4); %rows: tissue types, columns: type of optical property (mua, mus, g, n) 

		% epidermis 
		optical_data(1, 1) = (227*melanin(lambda) + 0.002*mua_blood(0.5, lambda) + (1-0.01-0.002)*25)*0.01;
    		optical_data(1, 2) = musr(lambda)/(1-g(lambda))*0.01;
		optical_data(1, 3) = g(lambda);
		optical_data(1, 4) = 1.4;

    		% papillary dermis
		bvf = 0.01;
		optical_data(2, 1) = (bvf*mua_blood(0.5, lambda) + (1-bvf)*25)*0.01;
    		optical_data(2, 2) = musr(lambda)/(1-g(lambda))*0.01;
		optical_data(2, 3) = g(lambda);
		optical_data(2, 4) = 1.4;
   
		% reticular dermis
		bvf = 0.01;
		optical_data(3, 1) = (bvf*mua_blood(0.8, lambda) + (1-bvf)*25)*0.01;
		optical_data(3, 2) = musr(lambda)/(1-g(lambda))*0.01;
		optical_data(3, 3) = g(lambda);
		optical_data(3, 4) = 1.4;
       
		%write to file
		filename = [name, num2str(floor(lam(i)))];    
		write_optical_props_file(filename, optical_data);
	end
end
