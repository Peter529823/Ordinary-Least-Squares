function [b,sterr,NWsterr,Rsquare,adjRsquare,Fstat,e] = OLS(y,X,constant,R,r)
%OLS simple OLS regression

%   y = T X 1 vector of dependent variable
%   X = T X K matrix of independent variables
%   constant = 0: no constant to be added;
%            = 1: constant term to be added (Default = 1)
%   R and r: inputs for H0: Rb=r
%       R has size J X K, r has size J X 1


%   b -- usual OLS estimate
%   sterr -- the standard error of b (assuming spherical errors)
%   NWsterr -- Newey West standard error of b
%   Rsquare -- the coefficient of determination
%   adjRsquare -- Rsquare adjusted for number of explanatory vars
%   Fstat -- for testing J linear restrictions on b (H0: Rb=r)
%       has J and T - K degrees of freedom
%   e -- the realized errors (is T X 1 vector)

NWlags=0; %to use default value -- Newey-West (1994) plug-in procedure

T=size(X,1);
if nargin<3, constant=1; end
if constant
    X=[ones(T,1),X];
end
K=size(X,2);

invXX=inv(X'*X);
b=invXX*(X'*y);
e=y-X*b;
s2=(e'*e)/(T-K);
varb=s2*invXX;
sterr=sqrt(diag(varb));
Rsquare=1-var(e)/var(y);
adjRsquare=1-(T-1)/(T-K)*(1-Rsquare);


if NWlags<=0
    NWlags = floor(4*((T/100)^(2/9)));
end
NWsterr = NeweyWest(e,X,T,K,NWlags);

    function y = NeweyWest(e,X,T,k,L)
        Q = 0;
        for l = 0:L
            w_l = 1-l/(L+1);
            for t = l+1:T
                if (l==0)   % This calculates the S_0 portion
                    Q = Q  + e(t) ^2 * X(t, :)' * X(t,:);
                else        % This calculates the off-diagonal terms
                    Q = Q + w_l * e(t) * e(t-l)* ...
                        (X(t, :)' * X(t-l,:) + X(t-l, :)' * X(t,:));
                end
            end
        end
        Q = (1/(T-k)) .*Q;

        y = sqrt(diag(T.*((X'*X)\Q/(X'*X))));

    end

Fstat=0;
if nargin>3
    J=numel(r);
    Fstat=1/J*(R*b-r)'*((R*varb*R')\(R*b-r));
end


   
    

end

