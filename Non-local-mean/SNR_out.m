%%% Calcul du SNR out  
%%%
%%% usage : y=snr_out(u,x) 
%%% 
%%% Entree:
%%% u: image d'entree
%%% x: image de reference
%%% NB: u et x doivent avoir la même dimension
%%% Sortie:
%%% y : valeur du SNR_out
%%% 

function y=SNR_out(u,x)

% conversion des tableaux en des vecteur 1D
tab = size(u);
tab_u = reshape(u,1,tab(1)*tab(2));
tab_x = reshape(x,1,tab(1)*tab(2));


% Variance de x
VAR_x = var(tab_x);

% Ecart quadratique
Eout = var(tab_x-tab_u);

% Calcul du SNRout

y = 10 * log(VAR_x/Eout) / log(10);



