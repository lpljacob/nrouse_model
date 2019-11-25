
%--------------------------------------------------------------------------
%                           Parameter setup
%-------------------------------------------------------------------------- 

PrimeDur = [50, 400];
TargetDur = 70;
MaskDur = 500 - TargetDur;
TestDur = 700;

feedback = 0.25; % strength of feedback from lexicosemantic layer to orthographic layer

% visual feature nodes coding
f_prime = 1;
f_target = 2;
f_mask = 3;
f_foil = 4; % only present when test word is different from target word

% weight matrices connecting visual features layer to orthographic layer
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

perceptual = zeros(1,max_timepoints+50,8); % P100/N170 predictions
memory = zeros(1,max_timepoints+250,8); % N400 predictions
residual = nan(1,8); % response-locked ERP predictions

save_output = zeros(1,max_timepoints,8);
save_resources = zeros(1,max_timepoints,8);


LinTypes = {'-k', '--r', '-k', '--r', '-r', '--k', '-r', '--k'};
conditionsNames = {'A-A-A','A-B-A',...
                        'AA-A-A','AA-B-A',...
                        'A-B-B', 'A-A-B',...
                        'AA-B-B','AA-A-B'};
                    
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

memory_contam = memory(:,1:end-250,:) + 0.3 .* perceptual(:,1:end-50,:); % N400 predictions with contamination from P100/N170