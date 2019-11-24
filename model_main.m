
%--------------------------------------------------------------------------
%                           Parameter setup
%-------------------------------------------------------------------------- 

PrimeDur = [50, 400];
TargetDur = 70;
MaskDur = 500 - TargetDur;
TestDur = 700;

feedback = 0.25;

% visual feature nodes coding
f_prime = 1;
f_target = 2;
f_mask = 3;
f_foil = 4; % only present when test word is different from target word

% weight matrices
feat_to_letter = [0,0;  % prime - varies by condition
                  1,0;  % target
                  0,0;  % mask
                  0,1]; % foil (when present)         
connection = eye(2);

% timing setup
max_timepoints = PrimeDur(2)+TargetDur+MaskDur+TestDur;

% generate empty matrices
feature_memb_matrix = zeros(4,max_timepoints,4);
feature_resources_matrix = zeros(4,max_timepoints,4);
feature_output_matrix = zeros(4,max_timepoints,4);
letter_memb_matrix = zeros(2,max_timepoints,4);
letter_resources_matrix = zeros(2,max_timepoints,4);
letter_output_matrix = zeros(2,max_timepoints,4);
word_memb_matrix = zeros(2,max_timepoints,4);
word_resources_matrix = zeros(2,max_timepoints,4);
word_output_matrix = zeros(2,max_timepoints,4);
storage_memb_matrix = zeros(2,max_timepoints,4);
storage_resources_matrix = zeros(2,max_timepoints,4);
storage_output_matrix = zeros(2,max_timepoints,4);

LinTypes = {'-k', '--r', '-k', '--r', '-r', '--k', '-r', '--k'};
conditionsNames = {'A-A-A','A-B-A',...
                        'AA-A-A','AA-B-A',...
                        'A-B-B', 'A-A-B',...
                        'AA-B-B','AA-A-B'};
                    
perceptual = zeros(1,max_timepoints+50,8);
memory = zeros(1,max_timepoints+250,8);
residual = nan(1,8);

save_output = zeros(1,max_timepoints,8);
save_resources = zeros(1,max_timepoints,8);

%--------------------------------------------------------------------------
%                            Simulation loop
%--------------------------------------------------------------------------

for i = 1:8
    
    % ---------------------- CONDITIONS SETUP ---------------------- %

    if i==1  % A A A - short test primed (same)
        prime_d=1;
        primed_tar=1;
        same_diff=1;
        LinType='-k';
    elseif i==2 % A B A - short test primed (diff)
        prime_d=1;
        primed_tar=2;
        same_diff=2;
        LinType='--k';
    elseif i==3 % AA A A - long test primed (same)
        prime_d=2;
        primed_tar=1;
        same_diff=1;
        LinType='-r';
    elseif i==4 % AA B A - long test primed (diff)
        prime_d=2;
        primed_tar=2;
        same_diff=2;
        LinType='--r';
    elseif i==5  % A B B - short test unprimed  (same)
        prime_d=1;
        primed_tar=2;
        same_diff=1;
        LinType='-b';
    elseif i==6 % A A B - short test unprimed  (diff)
        prime_d=1;
        primed_tar=1;
        same_diff=2;
        LinType='--b';
    elseif i==7 % AA B B - long test unprimed  (same)
        prime_d=2;
        primed_tar=2;
        same_diff=1;
        LinType='-g';
    elseif i==8 % AA A B - long test unprimed  (diff)
        prime_d=2;
        primed_tar=1;
        same_diff=2;
        LinType='--g';
    end
    
    % determine how visual layer connects to orthographic layer
    if primed_tar==1
        feat_to_letter(1,:) = [2,0];
    elseif primed_tar==2
        feat_to_letter(1,:) = [0,2];
    end

    timepoints = PrimeDur(prime_d)+TargetDur+MaskDur+TestDur;
    if prime_d==1
        primeonset = PrimeDur(2) - PrimeDur(1);
    else
        primeonset = 0;
    end
    
    % create empty nodes
    feature_memb = zeros(1,4);
    feature_resources = ones(1,4);
    feature_output = zeros(1,4);
    letter_memb = zeros(1,2);
    letter_resources = ones(1,2);
    letter_output = zeros(1,2);
    word_memb = zeros(1,2);
    word_resources = ones(1,2);
    word_output = zeros(1,2);
    storage_memb = zeros(1,2);
    storage_resources = ones(1,2);
    storage_output = zeros(1,2);
    
    % ------------------------- TRIAL LOOP ------------------------- %

    for t = 1:timepoints

        % determine word input to visual layer
        feature_input = zeros(1,4);
        if t <= PrimeDur(prime_d) % present prime
            feature_input(f_prime) = 1;
        elseif t > PrimeDur(prime_d) && t <= PrimeDur(prime_d)+TargetDur % present target
            feature_input(f_target) = 1;
        elseif t > PrimeDur(prime_d)+TargetDur && t <= PrimeDur(prime_d)+TargetDur+MaskDur % present mask
            feature_input(f_mask) = 1;
        elseif t > PrimeDur(prime_d)+TargetDur+MaskDur % present test
            if same_diff==1
                feature_input(f_target) = 1;
                res_ind = 1;
            else
                feature_input(f_foil) = 1;      
                res_ind = 2;
            end                        
        end
        
        % update visual layer
        [feature_output, feature_memb, feature_resources] = update_nrouse(feature_memb, feature_resources, ...
            feature_input, 1);
        
        % update orthographic layer
        letter_input = feature_output * feat_to_letter + feedback .* word_output * connection;
        [letter_output, letter_memb, letter_resources] = update_nrouse(letter_memb, letter_resources, ...
            letter_input, 2);

        % update semantic layer
        word_input = letter_output * connection;
        [word_output, word_memb, word_resources] = update_nrouse(word_memb, word_resources, word_input, 3);
        
        % update novelty layer
        storage_input = word_output * connection;
        [storage_output, storage_memb, storage_resources] = update_nrouse(storage_memb, storage_resources, ...
            storage_input, 4);
       
        
        % save values
        feature_output_matrix(:,t,i) = feature_output;
        feature_memb_matrix(:,t,i) = feature_memb;       
        feature_resources_matrix(:,t,i) = feature_resources;     
        letter_output_matrix(:,t,i) = letter_output;
        letter_memb_matrix(:,t,i) = letter_memb;
        letter_resources_matrix(:,t,i) = letter_resources;
        word_output_matrix(:,t,i) = word_output;
        word_memb_matrix(:,t,i) = word_memb;
        word_resources_matrix(:,t,i) = word_resources;     
        storage_output_matrix(:,t,i) = storage_output;
        storage_memb_matrix(:,t,i) = storage_memb;
        storage_resources_matrix(:,t,i) = storage_resources;  
        
    end  
    
    perceptual(:, 51+primeonset:end, i) = sum(letter_output_matrix(:,1:end-primeonset,i),1) - 3 * sum(word_output_matrix(:,1:end-primeonset,i),1);
    memory(:, 251+primeonset:end, i) = -1 * sum(storage_output_matrix(:,1:end-primeonset,i),1);
    
    tmp = PrimeDur(prime_d)+TargetDur+MaskDur;
    residual(i) = min(storage_output_matrix(res_ind,tmp:tmp+500,i));
    
    save_output(:, 1+primeonset:end, i) = storage_output_matrix(res_ind,1:end-primeonset,i);
    save_resources(:, 1+primeonset:end, i) = storage_resources_matrix(res_ind,1:end-primeonset,i);
  
end

memory_contam = memory(:,1:end-250,:) + 0.3 .* perceptual(:,1:end-50,:);

 % ------------------- PLOT FIGURES ------------------- %

 figure1 = figure; 

%     subplot(2,1,1);
    title('Perceptual ERP predictions (short prime');
    for i = [1,2,5,6]
        plot(perceptual(:,1:end-50,i),LinTypes{i},'LineWidth', 1.5,'DisplayName',conditionsNames{i});
        line([350 350], get(gca, 'ylim'));
        line([400 400], get(gca, 'ylim'));
        line([900 900], get(gca, 'ylim'));
        hold on
    end
    hold on
    
    subplot(2,1,2);
    title('Perceptual ERP predictions (long prime');
    for i = [3,4,7,8]
        plot(perceptual(:,1:end-50,i),LinTypes{i},'LineWidth', 1.5,'DisplayName',conditionsNames{i});
        line([400 400], get(gca, 'ylim'));
        line([900 900], get(gca, 'ylim'));
        hold on
    end
    hold on
    
    
    
    
    
figure(2);

    subplot(2,1,1);
    title('N400 ERP predictions (short prime');
    for i = [1,2,5,6]
        plot(memory(:,1:end-250,i),LinTypes{i},'LineWidth', 1.5,'DisplayName',conditionsNames{i});
        line([350 350], get(gca, 'ylim'));
        line([400 400], get(gca, 'ylim'));
        line([900 900], get(gca, 'ylim'));
        hold on
    end
    hold on
    
    subplot(2,1,2);
    title('N400 ERP predictions (long prime');
    for i = [3,4,7,8]
        plot(memory(:,1:end-250,i),LinTypes{i},'LineWidth', 1.5,'DisplayName',conditionsNames{i});
        line([400 400], get(gca, 'ylim'));
        line([900 900], get(gca, 'ylim'));
        hold on
    end
    hold on
 
leg = legend('show');
leg.FontSize = 12;

hold off

modelp100s = reshape(perceptual(:,1:end-150,[5,2,1,6]), 1500, 4);
modelp100l = reshape(perceptual(:,1:end-150,[7,4,3,8]), 1500, 4);

meanshort = mean(modelp100s,2);
meanlong = mean(modelp100l,2);

mp100s = modelp100s-meanshort;
mp100l = modelp100l-meanlong;

waveforms_plotting('short', mp100s, 'model', 'P100/N170 effects predictions, short prime duration', [-0.15 0.15])
waveforms_plotting2('short', meanshort, 'model', 'average P100/N170 predictions, short prime duration', [-0.4 0.4])
waveforms_plotting('long', mp100l, 'model', 'P100/N170 effects predictions, long prime duration', [-0.15 0.15])
waveforms_plotting2('long', meanlong, 'model', 'average P100/N170 predictions, long prime duration', [-0.4 0.4])

modeln400s = reshape(memory_contam(:,1:end-100,[5,2,1,6]), 1500, 4);
modeln400l = reshape(memory_contam(:,1:end-100,[7,4,3,8]), 1500, 4);

meanshort = mean(modeln400s,2);
meanlong = mean(modeln400l,2);

mn400s = modeln400s-meanshort;
mn400l = modeln400l-meanlong;

waveforms_plotting('short', mn400s, 'model', 'N400 effects predictions, short prime duration', [-0.2 0.2])
waveforms_plotting2('short', meanshort, 'model', 'average N400 predictions, short prime duration')
waveforms_plotting('long', mn400l, 'model', 'N400 effects predictions, long prime duration', [-0.2 0.2])
waveforms_plotting2('long', meanlong, 'model', 'average N400 predictions, long prime duration')


datan400s = n400(1:1500, [5,2,1,6]);
datan400l = n400(1:1500, [7,4,3,8]);
meanshort = mean(datan400s,2);
meanlong = mean(datan400l,2);

n400s = datan400s-meanshort;
n400l = datan400l-meanlong;

waveforms_plotting2('short', meanshort, 'data', 'average N400, short prime duration', [-1.6 1.3])
waveforms_plotting('short', n400s, 'data', 'N400 effects, short prime duration', [-1.6 1.3])
waveforms_plotting2('long', meanlong, 'data', 'average N400, long prime duration', [-1.6 1.3])
waveforms_plotting('long', n400l, 'data', 'N400 effects, long prime duration', [-1.6 1.3])

datap100s = reshape(tmp(:,1:1500, [6,2,1,5]), 1500, 4);
datap100l = reshape(tmp(:,1:1500, [8,4,3,7]), 1500, 4);
meanshort = mean(datap100s,2);
meanlong = mean(datap100l,2);

p100s = datap100s-meanshort;
p100l = datap100l-meanlong;

waveforms_plotting2('short', meanshort, 'data', 'average P100/N170, short prime duration', [-4 6])
waveforms_plotting('short', p100s, 'data', 'P100/N170 effects, short prime duration', [-1.1 1])
waveforms_plotting2('long', meanlong, 'data', 'average P100/N170, long prime duration', [-4 6])
waveforms_plotting('long', p100l, 'data', 'P100/N170 effects, long prime duration', [-1.1 1])

% activation and resources

outres = save_output(:,1:end-100,:);
outres = [outres; save_resources(:,1:end-100,:)];

titles = {'A-A-A';'A-B-A';'AA-A-A';'AA-B-A';'A-B-B';'A-A-B';'AA-B-B';'AA-A-B'};

figure1 = figure;
for i=1:8
    tmp = 'long';
    if ismember(i,[1 , 2 , 5 , 6]);
        tmp = 'short';
    end
    waveforms_plotting3(tmp, outres(:,:,i)', 'node', titles{i}, i)
end


figure1 = figure;
tmp = [zeros(350,4); reshape(sum(letter_output_matrix(:,1:end-primeonset-450,[5,2,1,6]),1), 1150, 4)];
waveforms_subplotting('short', tmp, 'layer', 'Visual objects layer output, short prime duration', 5, 3)
waveforms_subplotting('long', reshape(sum(letter_output_matrix(:,1:end-primeonset-100,[7,4,3,8]),1), 1500, 4), 'layer', 'Visual objects layer output, long prime duration', 6,3)

waveforms_subplotting('short', [zeros(350,4); reshape(sum(word_output_matrix(:,1:end-primeonset-450,[5,2,1,6]),1), 1150, 4)], 'layer', 'Lexical entries layer output, short prime duration', 3,3)
waveforms_subplotting('long', reshape(sum(word_output_matrix(:,1:end-primeonset-100,[7,4,3,8]),1), 1500, 4), 'layer', 'Lexical entries layer output, long prime duration',4,3)

waveforms_subplotting('short', [zeros(350,4); reshape(sum(storage_output_matrix(:,1:end-primeonset-450,[5,2,1,6]),1), 1150, 4)], 'layer', 'Maintained semantics layer output, short prime duration',1,3)
waveforms_subplotting('long', reshape(sum(storage_output_matrix(:,1:end-primeonset-100,[7,4,3,8]),1), 1500, 4), 'layer', 'Maintained semantics layer output, long prime duration',2,3)


% test
j=1;
for i=[1,2,5,6,3,4,7,8]
    minlistmem(j) = min(memory(:, end-650:end, i));
    perceptlist(j) = max(perceptual(:, 1000:end, i)) - min(perceptual(:, 1000:end, i));
    j = j+1;
end
minlistmem
perceptlist

for i=1:4
    j = (i-1)*2+1;
    priming_dur(i) = mean(perceptlist(j:j+1));
end
priming_dur

minlistmem = zeros(8,1);
j=1
for i=[1,2,5,6,3,4,7,8]
    minlistmem(j) = min(memory_contam(:, end-400:end, i));
%     minlistmem(j) = mean(memory_contam(:, end-400:end, i));
    j = j+1;
end
minlistmem

% n400 target
j=1;
for i=[2,1,4,3]
    if i == 2 || i == 4
        [m, ind] = min(memory(:, :, i));
        minlist(j) = m;
    else
        minlist(j) = memory(:, ind, i);
    end
    j = j+1;
end
minlist([2 1 4 3])











    
