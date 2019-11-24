function [output, memb, resources] = update_nrouse(memb, resources, input, layer)

inhibition = 0.9844; % self-inhibition
threshold = 0.15; % firing threshold
leak = 0.15; % leak conductance
depletion = 0.324; % depletion
recovery = 0.022; % recovery
integration = [.0294, .0609, .015, 0.3964]; % rate of integration

if layer==4 
    leak = 0.0103;
    inhibition = 0;
    depletion = 0.1036;
end

% output update, eq 7
output = (memb - threshold) .* resources;
output(output<0) = 0;
output(output>1) = 1;

% membrane potential update, eq 8
deltav = ((1 - memb) .* input - ...
    memb .* (leak + inhibition * sum(output))) .* integration(layer);
memb = memb + deltav;
memb(memb<0) = 0;
memb(memb>1) = 1;

% synaptic resources, eq 9
deltaa = (recovery * (1 - resources) - depletion .* output) .* integration(layer);
resources = resources + deltaa;
resources(resources<0) = 0;
resources(resources>1) = 1;
end
