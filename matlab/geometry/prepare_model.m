%this serves as a non-working example, since prepare_geometry uses tissue types not present in prepare_optprops, but hopefully you can in any case modify this yourself. 
lambdas = [400:850];
prepare_optprops(lambdas, 'examplemodel_optprops');
prepare_geometry('examplemodel_geometry');
