clc
clear all
data = load('iris2.data'); % Užkraunami duomenys
true_class = data(:, end); % Reali augalo klasė
parameters = data(:,1:end-1); % Taurėlapio bei žiedlapio parametrai
fis = readfis('iris.fis'); % Nuskaitomos narystės funkcijos
res = evalfis(fis, [parameters(:, 1) parameters(:, 2) parameters(:, 3) parameters(:, 4)]); % Vertinamas fuzzy sistemos tikslumas

% Pagal evalfis rezultatus nustatoma spėjama augalo klasė
predicted_class = [];
for x=1:length(res)
    if res(x) > 0 && res(x) < 1
        predicted_class = [predicted_class, 1];
    elseif res(x) > 1 && res(x) < 2
        predicted_class = [predicted_class, 2];
    elseif res(x) > 2 && res(x) < 3
        predicted_class = [predicted_class, 3];
    end
end
predicted_class = transpose(predicted_class); % Masyvo formatas suvienodinamas su true_class masyvo formatu (150x1 double)

% Skaičiuojami klasifikatoriaus vertinimo parametrai
[TP,TN,FP,FN] = deal(0);
P = length(res);
N = 2 * P;
for x=1:length(res)
    if predicted_class(x) == true_class(x)
        TP = TP + 1;
        TN = TN + 2;
    else
        FP = FP + 1;
        FN = FN + 1;
    end
end

% Skaičiuojamos klasifikavimo metrikos
TP
TN
FP
FN
P
N
accuracy = (TP+TN)/(P+N)
sensitivity = TP/P
precision = TP/(TP+FP)
matthew = (TP*TN - FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))