function varargout  = gplhr( sigma, X0, A, varargin )
% "The Generalized Preconditioned Locally Harmonic Residual (GPLHR) method
% for solving standard or generalized non-Hermitian partial eigenvalue problems." 
% Copyright (c) 2017, The Regents of the University of California, through Lawrence
% Berkeley National Laboratory (subject to receipt of any required approvals from 
% the U.S. Dept. of Energy). All rights reserved.
%
% Version  1.0
% Date:    01/20/2017
% Author:  Eugene Vecharynski
%
% GPLHR computes a number of eigenpairs of a matrix A or, in the case of
% the generalized eigenproblem, of a matrix pair (A,B), that correspond to the
% eigenvalues closest to a given shift value SIGMA. It can also return the partial 
% Schur decomposition of A or the generalized Schur decomposition of (A,B).
% The shift SIGMA can point to any location of the complex plane. 
%
% GPLHR is best suited for cases where matrices A and B are sparse or given 
% implicitly through their multiplication with a block of vectors, and where an 
% efficient preconditioner is available.
%
% The preconditioner is typically given by some approximation of the
% shift-and-invert operator inv(A - SIGMA*B), and can be obtained, e.g., 
% through an incomplete factorization of A - SIGMA*B or an approximate 
% solve of the linear system (A - SIGMA*B)W = Z. While it is possible 
% that GPLHR converges without preconditioning, the presence of the 
% preconditioner is essential for a rapid and robust convergence and is
% thus recommended.
%
% Please use the following reference to cite this code:
%
%    E. Vecharynski, C. Yang, and F. Xue: "Generalized preconditioned
%    locally harmonic residual method for non-Hermitian eigenproblems",
%    SIAM J. Sci. Comput., Vol. 38, Issue 1, pp. A500–A527, 2016
%
% Function usage:
%
%   LAMBDA = gplhr( SIGMA, X0, A ) returns a vector of nev eigenvalues of
%   a square matrix A that are closest to a given shift value SIGMA. The
%   matrix X0 has nev columns that contain initial guesses of eigenvectors.
%   The matrix A is either numeric or given by a handle of a function that
%   performs multiplication of A with a block of vectors.
%
%   LAMBDA = gplhr( SIGMA, X0, A, B ) returns a vector of nev eigenvalues of
%   the matrix pair (A,B) that are closest to a given shift value SIGMA.
%   In other words, solves the generalized eigenvalue problem Av=lambda*Bv.
%   If B=[], then a standard eigenvalue problem is solved. The matrix B is
%   either numeric or given by a handle of a function that performs
%   multiplication of B with a block of vectors.
%
%   LAMBDA = gplhr( SIGMA, X0, A, B, PREC ) also uses a preconditioner PREC
%   to accelearte the convergence (recommended).  The preconditioner PREC is
%   either numeric or given by a handle of a function that applies preconditioner
%   to a block of vectors.
%
%   LAMBDA = gplhr( SIGMA, X0, A, B, PREC, OPTS ) specifies the eigensolver's option
%   parameters OPTS (see description below).
%
%   [X, LAMBDA] = gplhr( SIGMA, X0, A, B, PREC, OPTS ) also returns the matrix X
%   of eigenvectors corresponding to the eigenvalues LAMBDA. 
%
%   [X, LAMBDA, FLAG] = gplhr( SIGMA, X0, A, B, PREC, OPTS ) also returns the failure
%   flag. FLAG==0 indicates that the wanted eigenpairs successfully converged to the
%   desired tolerance level. Otherwise, FLAG==1.
%
%   [X, LAMBDA, FLAG, V] = gplhr( SIGMA, X0, A, B, PREC, OPTS ) also returns an 
%   orthonormal matrix V of right Schur vectors. This is an orthonormal basis 
%   of the invariant subspace associated with LAMBDA. 
%
%   [X, LAMBDA, FLAG, V, Q] = gplhr( SIGMA, X0, A, B, PREC, OPTS ) also returns 
%   an orthonormal matrix Q of left Schur vectors. For standard eigenproblem, Q = V. 
%   
%   [X, LAMBDA, FLAG, V, Q, S] = gplhr( SIGMA, X0, A, B, PREC, OPTS ) also returns 
%   an upper triangular matrix S from the generalized Schur decomposition (A*V = Q*S
%   and B*V = Q*T) of the matrix pair (A,B). For standard eigenproblem, S is the Schur 
%   form of A and A*V = V*S.
%   
%   [X, LAMBDA, FLAG, V, Q, S, T] = gplhr( SIGMA, X0, A, B, PREC, OPTS ) also returns 
%   an upper triangular matrix T from the generalized Schur decomposition of (A,B). 
%   For standard eigenproblem, T is nev-by-nev identity matrix.
%
%   [X, LAMBDA, FLAG, V, Q, S, T, RESHIST] = gplhr( SIGMA, X0, A, B, PREC, OPTS ) 
%   also returns the (iter+1)-by-nev history matrix RESHIST of relative residual 
%   norms in all iterations, where iter is the number of iterations performed by 
%   GPLHR to achieve convegence. RESHIST(i+1,j) contains a relative residual norm 
%   of j-th eigenpair at iteration i. RESHIST(1,:) contains relative residual norms
%   of the initial guesses.
%
%   The GPLHR option parameters OPTS:
%       OPTS.tol:    convergence tolerance ( by default, OPTS.tol=1e-8 ).
%                    Convergence of the i-th pair is achieved when the corresponding 
%                    Schur residual is less than tol*(norm(A,2) + theta_i*norm(B,2)),
%                    where theta_i is the i-th eigenvalue approximation
%       OPTS.maxit:  maximum number of iterations ( by default, OPTS.maxit = 50 )
%       OPTS.m:      number of S-blocks to build the (m+3)*nev GPLHR search
%                    subspace ( by default, OPTS.m = 1 )
%                    Increasing OPTS.m can be used as means to improve/stabilize
%                    convergence. However, larger values of OPTS.m yield a larger
%                    number of matrix-vector products and preconditioning operations
%                    per GPLHR iteration and require more memory. Recommended values for 
%                    OPTS.m are somewhere between 1 and 5.
%       OPTS.issym:  if true, indicates that the targeted eigenproblem is symmetric/Hermitian,
%                    i.e., A = A' and B = B' and B is symmetric/Hermitian positive definite.
%                    This option does not change the algorithm flow and only ensures that, 
%                    on return, eigenvalues in lambda are real and eigenvectors in X are 
%                    orthonormal, or B-orthonormal, in the 
%                    case of generalized eigenproblem ( by default, OPTS.issym = false )
%       OPTS.verbose: verbosity level ( by default, OPTS.verbose=0 )
%                    OPTS.verbose == 0, no output
%                    OPTS.verbose == 1, print limited info on GPLHR progress
%                    OPTS.verbose > 1,  print full convergence info
%
%   Example:
%      clear all; clc; 
%      load west0479;
%      A = west0479;
%      sigma = 0.0;
%      n = length(A);
%      nev = 10;
%      [L,U,P] = ilu(A-sigma*speye(n), struct('type','ilutp','droptol',1e-6));
%      prec = @(x) (U\(L\(P*x)));
%      X0 = randn(n,nev);
%      opts.tol = eps; opts.m = 2; opts.verbose = 2; 
%      [X, lambda, flag, V, ~, S, ~, reshist] = gplhr(sigma, X0, A, [], prec, opts);
%      semilogy(max(reshist,[],2)); hold on;
%      set(gca,'FontSize',16,'FontWeight','bold');
%      title('GPLHR convergence for west0479', 'fontsize',16)
%      ylabel('Largest rel. residual norm','fontsize',16);
%      xlabel('Iteration number','fontsize',16);
%      drawnow;
%
%Redistribution and use in source and binary forms, with or without
%modification, are permitted provided that the following conditions are
%met:
%(1) Redistributions of source code must retain the above copyright
%notice, this list of conditions and the following disclaimer.
%(2) Redistributions in binary form must reproduce the above copyright
%notice, this list of conditions and the following disclaimer in the
%documentation and/or other materials provided with the distribution.
%(3) Neither the name of the University of California, Lawrence
%Berkeley National Laboratory, U.S. Dept. of Energy nor the names of
%its contributors may be used to endorse or promote products derived
%from this software without specific prior written permission.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
%"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
%LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
%A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
%OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
%SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
%LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
%THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%You are under no obligation whatsoever to provide any bug fixes,
%patches, or upgrades to the features, functionality or performance of
%the source code ("Enhancements") to anyone; however, if you choose to
%make your Enhancements available either publicly, or directly to
%Lawrence Berkeley National Laboratory, without imposing a separate
%written license agreement for such Enhancements, then you hereby grant
%the following license: a non-exclusive, royalty-free perpetual license
%to install, use, modify, prepare derivative works, incorporate into
%other computer software, distribute, and sublicense such enhancements
%or derivative works thereof, in binary and source code form.

% Input validation and initializations

is_gen = false;       % generalized eigenproblem
has_prec = false;     % has preconsditioner
opts = [];

% check number of inputs and outputs

if ( nargin < 3 )
    error( 'GPLHR:TooFewInputs', ...
        'There must be at least 3 input arguments: sigma, X0, and A' );
end

if ( nargin > 6 )
    error( 'GPLHR:TooManyInputs', ...
        'There must be at most 6 input arguments: sigma, X0, A, B, prec, and opts' );
end

if ( nargout > 8 )
    error( 'GPLHR:TooManyOutputs', ...
        'There must be at most 8 outputs: LAMBDA, X, FLAG, V, Q, S, T, and RESHIST' );
end

% setup and validate solver option parameters

if ( nargin == 6 )
    opts = varargin{3};
    if ( ~isempty( opts ) && ~isstruct( opts ) )
        error( 'GPLHR:OptsWrongInput', 'Sixth argument must be a structure' );
    end
end

if ~isfield( opts, 'verbose' )
    opts.verbose = 0;
elseif ( ~isnumeric( opts.verbose ) || isempty( opts.verbose ) || opts.verbose<0 )
    error('GPLHR:VerboseWrongInput', 'opts.verbose must be a nonnegative integer' );
end
verbose = floor( opts.verbose );

if ( verbose > 0 )
    fprintf( '\n');
    disp( '******************************************* ');
    fprintf(  '***       Start GPLHR eigensolver       *** \n');
    disp( '******************************************* ');
    fprintf('\n');
end

if ~isfield( opts, 'tol' )
    opts.tol = 1e-8;
elseif (  ~isnumeric( opts.tol ) || isempty( opts.tol ) || opts.tol < 0 )
    error('GPLHR:ToleranceWrongInput', 'opts.tol must be a nonnegative number' );
end
tol = opts.tol;

if ~isfield( opts, 'maxit' )
    opts.maxit = 50;
elseif (  ~isnumeric( opts.maxit ) || isempty( opts.maxit ) || opts.maxit < 0 )
    error('GPLHR:MaxitWrongInput', 'opts.maxit must be a nonnegative integer' );
end
maxit = floor( opts.maxit );

if ~isfield( opts, 'm' )
    opts.m = 1;
elseif ( ~isnumeric( opts.m ) || isempty( opts.m ) || opts.m < 0 )
    error('GPLHR:MWrongInput', 'opts.m must be a nonnegative integer' );
end
m = floor( opts.m );

if ~isfield( opts, 'issym' )
    opts.issym = false;
elseif ( ~islogical( opts.issym ) || isempty( opts.issym ) )
    error('GPLHR:IssymWrongInput', 'opts.issym must be of logical type' );
end
issym = opts.issym;

% validate sigma

if isempty( sigma )
    error( 'GPLHR:SigmaEmptyInput', ...
        'First argument is empty' );
end

if ~isnumeric( sigma )
    error( 'GPLHR:SigmaNotNumericInput', ...
        'First argument must be of a numeric type' );
end

if ( verbose>0 )
    fprintf('Shift sigma = (%10.5f, %10.5f)...\n', real(sigma), imag(sigma) );
end

% validate initial guess

if isempty( X0 )
    error( 'GPLHR:InitGuessEmptyInput', 'Second argument is empty' );
end

if ~isnumeric( X0 )
    error( 'GPLHR:InitGuessNotNumericInput', ...
        'Second argument must be of a numeric type' );
end

[ n, nev ] = size( X0 );

if ( verbose>0 )
    fprintf('Detected initial guess of %d eigenvectors...\n', nev );
    fprintf('Problem size = %d...\n', n );
end

if ( nev > n )
    error( 'GPLHR:InitGuessFatInput',...
        'Second argument must have more rows than columns');
end

% validate operator A

if isempty( A )
    error( 'GPLHR:OperatorAEmptyInput', 'Third argument is empty' );
end

if ( ~isnumeric( A ) && ~isa( A, 'function_handle' ) )
    error( 'GPLHR:OperatorAWrongInput', ...
        'Third argument must be either numeric or given by a function handle' );
end

if isnumeric( A )
    [ n1, n2 ] = size( A );
    if ( n1 ~= n2 )
        error( 'GPLHR:MatrixANotSquareInput', 'Third argument must be a square matrix' );
    elseif ( n1 ~= n )
        error( 'GPLHR:WrongSizeInput', ...
            'Dimensions of the second and third arguments do not match' );
    end
end

% validate operator B

if ( nargin > 3 )
    B = varargin{1};
    if ( ~isempty( B ) )
        is_gen = true;
    end
    if ( ~isnumeric( B ) && ~isa( B, 'function_handle' ) )
        error( 'GPLHR:OperatorBWrongInput', ...
            'Fourth argument must be either numeric or given by a function handle' );
    end
    if ( isnumeric( B ) && ~isempty( B ) )
        [ n1, n2 ] = size( B );
        if ( n1 ~= n2 )
            error( 'GPLHR:MatrixBNotSquareInput', 'Fourth argument must be a square matrix' );
        elseif ( n1 ~= n )
            error( 'GPLHR:MatrixBWrongSizeInput', ...
                'Dimensions of the second and fourth arguments do not match' );
        end
    end
end

if ( verbose > 0 )
    if ( is_gen )
        disp('Generalized eigenvalue problem...');
    else
        disp('Standard eigenvalue problem...');
    end
end

% validate preconditioner

if ( nargin > 4 )
    PREC = varargin{2};
    if ( ~isempty( PREC ) )
        has_prec = true;
    end
    if ( ~isnumeric( PREC ) && ~isa( PREC, 'function_handle' ) )
        error( 'GPLHR:PreconditionerWrongInput', ...
            'Fifth argument must be either numeric or given by a function handle' );
    end
    if ( isnumeric( PREC ) && ~isempty( PREC ) )
        [ n1, n2 ] = size( PREC );
        if ( n1 ~= n2 )
            error( 'GPLHR:PrecinditionerNotSquareInput', ...
                'Fifth argument must be a square matrix' );
        elseif ( n1 ~= n )
            error( 'GPLHR:PreconditionerWrongSizeInput', ...
                'Dimensions of the second and fifth arguments do not match' );
        end
    end
end

if ( verbose > 0 )
    if ( has_prec )
        disp('Preconditioner provided...');
    else
        disp('No preconditioner provided...');
    end
end

% Setup problem operators: operA, operB, and operT

if isa( A,'function_handle' )
    disp('Operator A is given by a function handle...');
    operA = A;
else
    operA = @(X)( A*X );
    if ( verbose > 0 )
        if issparse( A )
            disp('Operator A is given by a sparse matrix...');
        else
            disp('Operator A is given by a dense matrix...');
        end
    end
end

if ( is_gen )
    if isa( B,'function_handle' )
        operB = B;
        if ( verbose>0 )
            disp('Operator B is given by a function handle...');
        end
    else
        operB = @(X)( B*X );
        if ( verbose>0 )
            if issparse( B )
                disp('Operator B is given by a sparse matrix...');
            else
                disp('Operator B is given by a dense matrix...');
            end
        end
    end
end

if ( has_prec )
    if isa( PREC,'function_handle' )
        if ( verbose>0 )
            disp('Preconditioner is given by a function handle...');
        end
        operT = PREC;
    else
        operT = @(X)( PREC*X );
        if ( verbose>0 )
            if issparse( PREC )
                disp('Preconditioner is given by a sparse matrix...');
            else
                disp('Preconditioner is given by a dense matrix...');
            end
        end
    end
else
    % preconditioner = I
    operT = @(X)( X );
end

% report solver option parameters
if ( verbose > 0 )
    fprintf( 'Requested convergence tolerance = %e...\n', tol );
    fprintf( 'Maximum number of iterations = %d...\n', maxit );
    fprintf( 'Number of S-blocks m = %d...\n', m );
    if ( issym )
        disp('Problem identified as symmetric/Hermitian (OPTS.issym=true)...');
    end	
end

% estimate 2-norms of A and B 
estnrmA = estnrm2( operA, n );
estnrmB = 1.0;        
if ( is_gen )
    estnrmB = estnrm2( operB, n );
end
if ( verbose > 0 )
    fprintf('Estimated 2-norm of operator A = %4.3e...\n', estnrmA );
    if ( is_gen )
        fprintf('Estimated 2-norm of operator B = %4.3e...\n', estnrmB );
    end	   
end	

% hard coded constants
m_max = max( 20, m );   % max number of S-blocks
force_mult_iter = 100;  % force extra multiplies with A and B
% at every force_mult_iter'th iteration

% allocate memory for S, A*S and B*S blocks
S  = zeros( n, m*nev );
AS = zeros( n, m*nev );
if ( is_gen )
    BS = zeros( n, m*nev );
end

if ( nargout == 8)
    reshist = zeros( maxit+1, nev );
end

% Pre-processing

[ V, ~ ] = qr( X0, 0 );
AV = operA( V );

if ( is_gen )
    BV = operB( V );
    Q = AV - sigma*BV;
    [ Q, ~ ] = qr( Q, 0 );
    M = Q'*BV;
else
    Q = AV - sigma*V;
    [ Q , ~ ] = qr( Q, 0 );
    M = Q'*V;
end
K = Q'*AV;

[ TRIUA, TRIUB, Y, X ] = GetOrdqz( K, M, sigma ) ;

V  = V*X;  % right Schur vectors
AV = AV*X;
Q  = Q*Y;  % left Schur vectors

[ MA, MB ] = GetTriu( TRIUA, TRIUB );

lambda  = GetLambda( diag( MA ), diag( MB ) );

if ( is_gen )
    BV = BV*X;
    W = AV*MB - BV*MA;
else
    W = AV*MB - V*MA;
end

relres = GetRelres( W, lambda, estnrmA, estnrmB );

[ nlock, nact ] = GetNumLockAct( relres, tol );

% Save history

if ( nargout == 8 )
    reshist( 1, : ) = relres;
end

% Report initial guess
if ( verbose > 1 )
    msg = '*** Initial guess: ***';
    print_progress( msg, lambda, relres, nlock );
end

if ( nlock  == nev )
    if ( verbose > 0 )
        fprintf('Initial guess is sufficiently accurate. Nothing to be done.\n');
    end
    cv_flag = true;
else
    cv_flag = false;
end

% Main loop

if ( verbose > 0 )
    fprintf('\nRunning GPLHR iterations ...\n');
end

iter = 1;

while ( iter <= maxit && ~cv_flag )
    
    [ V, R ] = qr( V, 0 );
    AV = AV / R;
    if ( is_gen )
        BV = BV / R;
    end
    
    [ Q, ~ ] = qr( Q, 0 );
    
    % compute preconditioned (projected) residual
    if ( is_gen )
        W( :, nlock+1:nev ) = W( :, nlock+1:nev ) - Q*( Q'*W( :, nlock+1:nev ) );
    else
        W( :, nlock+1 : nev ) = W( :, nlock+1:nev ) - V*( V'*W( :, nlock+1:nev ) );
    end
    W( :, nlock+1:nev ) = operT( W( :, nlock+1:nev ) );
    
    W( :, nlock+1:nev ) = W( :, nlock+1:nev ) - V*( V'*W( :, nlock+1:nev ) );
    if ( nev > 1 )
        % this line hangs if nev == 1. MATLAB bug?
        [ W( :, nlock+1:nev ), ~ ] = qr( W( :, nlock+1:nev ), 0 );
    else
        nrm = norm( W( :, nlock+1:nev ) );
        W( :, nlock+1:nev ) = W(:, nlock+1:nev )/nrm;
    end
    
    
    AW( :, nlock+1:nev ) = operA( W( :, nlock+1:nev ) ) ;
    if ( is_gen )
        BW( :, nlock+1:nev ) = operB( W( :, nlock+1:nev ) ) ;
    end
    
    % generate S-blocks
    
    ns = min( floor( m*nev/nact ), m_max );
    
    if ( ns > 0 )
        ends = ns*nact;
        if ( is_gen )
            S( : , 1 : nact ) = ...
                AW( :, nlock+1:nev )*MB( nlock+1:nev, nlock+1:nev ) - ...
                BW( :, nlock+1:nev )*MA( nlock+1:nev, nlock+1:nev );
        else
            S( :, 1:nact ) = ...
                AW( :, nlock+1:nev )*MB( nlock+1:nev, nlock+1:nev ) - ...
                W( :, nlock+1:nev )*MA( nlock+1:nev, nlock+1:nev );
        end
        
        if ( is_gen )
            S( :, 1:nact ) = S( :, 1:nact ) - Q*( Q'*S( :, 1:nact ) );
        else
            S( :, 1:nact ) = S( :, 1:nact ) - V*( V'*S( :, 1:nact ) );
        end
        S( :, 1:nact ) = operT( S( :, 1:nact ) );
        
        S( :, 1:nact ) = S( :, 1:nact ) - V*( V'*S( :, 1:nact ) );
        S( :, 1:nact ) = S( :, 1:nact ) - ...
            W( :, nlock+1:nev )*( W( :, nlock+1:nev )'*S( : , 1:nact ) );
        
        [ S( :, 1:nact ) , ~ ] = qr( S( :, 1:nact ), 0 );
        
        AS( :, 1:nact ) = operA( S( :, 1:nact ) );
        if ( is_gen )
            BS( :, 1:nact ) = operB( S( :, 1:nact ) );
        end
        
        for j = 2 : ns
            j1 = ( j - 1 )*nact + 1;
            j2 = j*nact;
            
            if ( is_gen )
                S( :, j1:j2 ) = ...
                    AS( :, j1-nact:j2-nact )*MB( nlock+1:end, nlock+1:end ) - ...
                    BS( :, j1-nact:j2-nact )*MA( nlock+1:end, nlock+1:end);
            else
                S( :, j1:j2 ) = ...
                    AS( :, j1-nact:j2-nact )*MB( nlock+1:end, nlock+1:end) - ...
                    S( :, j1-nact:j2-nact )*MA( nlock+1:end, nlock+1:end);
            end
            
            if ( is_gen )
                S( :, j1:j2 )  = S( : , j1:j2 ) - Q*( Q'*S( :, j1:j2 ) );
            else
                S( :, j1:j2 )  = S( : , j1:j2 ) - V*( V'*S( : , j1:j2 ) );
            end
            S( :, j1:j2 ) = operT( S( :, j1:j2 ) );
            
            S( : , j1:j2 ) = S( :, j1:j2 ) - V*( V'*S( :, j1:j2 ) );
            S( : , j1:j2 ) = S( :, j1:j2 ) - ...
                W( :, nlock+1:nev )*( W( : , nlock+1:nev )'*S( : , j1:j2 ) );
            S( :, j1:j2 ) = S( :, j1:j2 ) - ...
                S( :, 1:j1-1 )*( S( :, 1:j1-1 )'*S( :, j1:j2 ) );
            
            [S( :, j1:j2 ), ~ ] = qr( S( :, j1:j2 ), 0 );
            
            AS( : ,j1:j2 ) = operA( S( :, j1:j2 ) );
            if ( is_gen )
                BS( :, j1:j2 ) = operB( S( :, j1:j2 ) );
            end
        end
    end
    
    % orthogonalize 'conjugate' direction
    if ( iter > 1 )
        H  = V'*P;
        P  = P - V*H;
        AP = AP - AV*H;
        if ( is_gen )
            BP = BP - BV*H;
        end
        
        H = W( :, nlock+1:nev )'*P;
        P = P - W( :, nlock+1:nev )*H;
        AP = AP - AW( :, nlock+1:nev )*H;
        if ( is_gen )
            BP = BP - BW( :, nlock+1:nev )*H;
        end
        
        if ( ns > 0 )
            H = S( :, 1:ends )'*P;
            P = P - S( :, 1:ends )*H;
            AP = AP - AS( :, 1:ends )*H;
            if ( is_gen )
                BP = BP - BS( :, 1:ends )*H;
            end
        end
        
        [ P, R ] = qr( P, 0 );
        AP = AP / R;
        if ( is_gen )
            BP = BP / R;
        end
    else
        P = [];
        AP = [];
        if ( is_gen )
            BP = [];
        end
    end
    
    % setup projected problem with test subspace = [Q,Q1,Q2,Q3]
    if ( is_gen )
        [ Q1, Q2, Q3 ] = SetupTestSubsp( AW( :, nlock+1:end ), AS( :, 1:ends ), AP,...
            BW( :, nlock+1:end ), BS( :, 1:ends ), BP, Q, sigma );
        
        [ K, M ] = GetProjProb( AV, AW( :, nlock+1:end ), AS( :, 1:ends ), AP,...
            BV, BW( : , nlock+1:end ), BS( : , 1:ends ), BP, Q, Q1, Q2, Q3 );
        
    else
        [ Q1, Q2, Q3 ] = SetupTestSubsp( AW( :, nlock+1:end ), AS( :, 1:ends ), AP,...
            W( :, nlock+1:end ), S( :, 1:ends ), P, Q, sigma );
        
        [ K, M ] = GetProjProb( AV, AW( :, nlock+1:end ), AS( :, 1:ends ), AP,...
            V, W( :, nlock+1:end ), S( :, 1:ends ), P, Q, Q1, Q2, Q3 );
    end
    
    % harmonic Rayleigh-Ritz
    
    [ TRIUA, TRIUB, Y, X ] = GetOrdqz( K, M, sigma );
    
    TRIUA = TRIUA( 1:nev, 1:nev );
    TRIUB = TRIUB( 1:nev, 1:nev );
    
    % update V and P
    
    coordV = X( 1:nev, 1:nev );
    coordW = X( nev+1:nev+nact, 1:nev );
    coordS = X( nev+nact+1 : nev+(ns+1)*nact, 1:nev );
    
    p_coordV = X( 1:nev, nev+1:2*nev );
    p_coordW = X( nev+1:nev+nact, nev+1:2*nev );
    p_coordS = X( nev+nact+1:nev+(ns+1)*nact, nev+1:2*nev );
    
    if ( iter > 1 )
        coordP = X( nev+(ns+1)*nact+1:end, 1:nev );
        p_coordP = X( nev+(ns+1)*nact+1:end, nev+1:2*nev );
    end
    
    t = nact*ns;
    work = V;
    if ( iter > 1 )
        V = V*coordV + P*coordP;
        P = work*p_coordV + P*p_coordP;
        V = V +  W( :, nlock+1:end )*coordW + S( :, 1:t )*coordS;
        P = P +  W( :, nlock+1:end )*p_coordW + S( :, 1:t )*p_coordS;
    else
        V = V*coordV + W( :, nlock+1:end )*coordW + S( :, 1:t )*coordS;
        P = work*p_coordV + W( :, nlock+1:end )*p_coordW + S( :, 1:t )*p_coordS;
    end
    
    % update AV and AP
    
    work = AV;
    if ( iter > 1 )
        AV = AV*coordV + AP*coordP;
        AP = work*p_coordV + AP*p_coordP;
        AV = AV +  AW( :, nlock+1:end )*coordW + AS( :, 1:t )*coordS;
        AP = AP +  AW( :, nlock+1:end )*p_coordW + AS( :, 1:t )*p_coordS;
    else
        AV = AV*coordV + AW( :, nlock+1:end )*coordW + AS( :, 1:t )*coordS;
        AP = work*p_coordV + AW( : , nlock+1:end)*p_coordW + AS( :, 1:t )*p_coordS;
    end
    
    % update BV and BP
    
    if ( is_gen )
        work = BV;
        if ( iter > 1 )
            BV = BV*coordV + BP*coordP;
            BP = work*p_coordV + BP*p_coordP;
            BV = BV +  BW( :, nlock+1:end )*coordW + BS( :, 1:t )*coordS;
            BP = BP +  BW( :, nlock+1:end )*p_coordW + BS( :, 1:t )*p_coordS;
        else
            BV = BV*coordV + BW( :, nlock+1:end )*coordW + BS( :, 1:t )*coordS;
            BP = work*p_coordV + BW( :, nlock+1:end )*p_coordW + BS( :, 1:t )*p_coordS;
        end
    end
    
    % 'refresh' AV and BV after sufficiently large number of iters
    if ( mod( iter, force_mult_iter ) == 0 )
        AV = operA( V );
        if ( is_gen )
            BV = operB( V );
        end
    end
    
    % update Q
    
    coordV = Y( 1:nev, 1:nev );
    coordW = Y( nev+1:nev+nact, 1:nev );
    coordS = Y( nev+nact+1:nev+(ns+1)*nact, 1:nev );
    Q = Q*coordV + Q1*coordW + Q2*coordS;
    if ( iter > 1 )
        coordP = Y( nev+(ns+1)*nact+1:end, 1:nev );
        Q = Q + Q3*coordP;
    end
    
    % compute residual
    
    [ MA, MB ] = GetTriu( TRIUA, TRIUB );

    lambda  = GetLambda( diag( MA ), diag( MB ) );
    
    if ( is_gen )
        W  = AV*MB - BV*MA;
    else
        W = AV*MB - V*MA;
    end
    
    relres = GetRelres( W, lambda, estnrmA, estnrmB );
    
    % Store residual and eigenvalue histories
    
    [ nlock, nact ] = GetNumLockAct( relres, tol );
        
    if ( nargout == 8 )
        reshist( iter+1, : ) = relres;
    end
    
    % Print iteration info ...
    if ( verbose > 1 )
        msg = ['*** Iteration ', int2str(iter), ' ***'];
        print_progress( msg, lambda, relres, nlock );
    end
    
    if ( nlock  == nev )
        cv_flag = true;
        if ( verbose > 0 )
            fprintf('\n*** All eigenpairs converged to the desired tolerance ***\n');
        end
    end
    
    iter = iter + 1;
    
end

% Post-processing

if ( ~cv_flag && verbose > 0 )
    fprintf('\n*** Eigenpairs did not converged to the desired tolerance ***\n');
end

% return eigenvalues
[ V, ~ ] = qr( V, 0 );
[ Q, ~ ] = qr( Q, 0 );
AV = operA( V );
if ( is_gen )
    BV = operB( V );
end

if ( ~issym )
    % harmonic Rayl.-Ritz 	
    K = Q'*AV;
    if ( is_gen )
        M = Q'*BV;
    else
        M = Q'*V;
    end
else
    % stand. Rayl.-Ritz to preserve symmetry	
    K = V'*AV;
    if ( is_gen )
        M = V'*BV;
    else
        M = V'*V;
    end
    K = 0.5*( K + K' );
    M = 0.5*( M + M' );
    if ( is_gen && nargout > 4 )
	% will still need harmonic projection
        % to compute left Schur vectors of
        % generalized Hermitian eigenproblems 	
        Kharm = Q'*AV;  	    
        Mharm = Q'*BV;  	    
    end	    
end

[ Z, D ] =  eig( K, M );
[ ~ , idx ] = sort(abs( diag(D) - sigma ) );
lambda = diag( D( idx, idx ) );

if ( nargout < 2 )
   varargout{ 1 } = lambda;
end

% return eigenvectors and print final result
if ( nargout > 1 || verbose > 0 )

    X = V*Z( :, idx );

    if ( nargout > 1 )
        varargout{ 1 } = X ;
        varargout{ 2 } = lambda ;
    end

    if ( verbose > 0 )
        % compute final eigenresiduals
        AX = AV*Z( :, idx );
        if ( is_gen )
            BX = BV*Z( :, idx );
            for j = 1 : nev
                if ( abs(lambda(j)) ~= Inf )
                    W( :, j ) = AX( :, j ) - lambda(j)*BX( :, j);
                else
                    W( :, j ) = BX( :, j );
                end
            end
        else
            W = AX - X*diag( lambda );
        end
        relres = GetRelres( W, lambda, estnrmA, estnrmB );
        msg = '*** Final eigenvalues and relative eigenresidual norms: ***';
        print_progress( msg, lambda, relres, nlock );
        fprintf( 'Total number of iterations = %d\n', iter-1 );
    end
end

% return failure flag
if ( nargout > 2 )
    varargout{ 3 } = ~cv_flag;
end

% return right Schur vectors 
if ( nargout > 3 )
    if ( is_gen && issym && nargout > 4 )	
        [ TRIUA, TRIUB, Y, Z ] = GetOrdqz( Kharm, Mharm, sigma ) ;
    else 	
        [ TRIUA, TRIUB, Y, Z ] = GetOrdqz( K, M, sigma ) ;
    end
    if ( issym && ~is_gen )
        V = X;  % Schur vectors same as eigenvectors
    else	
        V  = V*Z;
    end
    varargout{ 4 } = V; 
end

% return left Schur vectors 
if ( nargout > 4 )
    if ( is_gen )    
        Q = Q*Y;
        varargout{ 5 } = Q;
    else
        varargout{ 5 } = V;
    end
end

% return triangular Schur factors
if ( nargout > 5 )
    if ( is_gen )    
        varargout{ 6 } = TRIUA;   
    elseif ( ~issym )
        varargout{ 6 } = TRIUB\TRIUA;
    else 
	% standard Hermitian eigenproblem 
        % has the diagonal Schur form	
        varargout{ 6 } = diag( lambda );
    end	
end

if ( nargout > 6 )
    if ( is_gen )    
        varargout{ 7 } = TRIUB; 
    else
        varargout{ 7 } = eye( nev );	
    end	
end   

% return residual history
if ( nargout == 8)
    varargout{ 8 } = reshist( 1 : iter, : );
end

if ( verbose > 0 )
    fprintf( '\n');
    disp( '******************************************* ');
    fprintf(  '***      Finish GPLHR eigensolver       *** \n');
    disp( '******************************************* ');
    fprintf('\n');
end

end % end gplhr



%----- Auxiliary functions   ------ %

function nrmest = estnrm2( operA, n )
% Cheap rough estimate of 2-norm of an 
% implicitly given matrix operA
nvec = 5; % number of sample vectors
X = randn( n, nvec );
nrmest = norm( operA(X), 'fro' ) / norm( X, 'fro' );
end


function lambda  = GetLambda( alpha, beta )
% Compute eigenvalues from generalized eigenvalues,
% LAMBDA(j) = ALPHA(j)/BETA(j).
% If beta(j) < 10*machine_eps then set LAMBDA(j) = Inf
nev = length( alpha );
lambda = zeros( nev, 1 );
MZERO = 10*eps;
for j = 1 : nev
    if ( abs( beta(j) ) >= MZERO)
        lambda( j ) = alpha( j ) / beta( j );
    elseif ( abs( alpha(j) ) >= MZERO)
        lambda( j ) = Inf;
    else
        error( 'GPLHR:IndeterminateEigenvalue', ...
            'Encountered indeterminate eigenvalue 0/0. The problem is ill-posed.  ' );
    end
end
end

function [ MA, MB ] = GetTriu( TRIUA, TRIUB )
% Given generalized Schur form (TRIUA, TRIUB),
% compute triangular factors MA and MB in the 'Q-free'
% Schur form AV*MB = BV*MA.
nev = size( TRIUA, 1 );
CA = eye( nev );
CB = eye( nev );
for j = 1 : nev
    if abs( TRIUA(j,j) ) >= abs( TRIUB(j,j) )
        CA( j, j ) = ( 1 - TRIUB(j,j) )/TRIUA(j,j);
    else
        CA(j,j) = 0;
        CB(j,j) = 1/( TRIUB(j,j) );
    end
end
G = TRIUA*CA + TRIUB*CB;
MA = ( CB/G )*TRIUA;
MB = speye( nev ) - ( CA/G )*TRIUA;
end

function [ TRIUA, TRIUB, Q, V ] = GetOrdqz( K, M, sigma )
% Compute generalized Schur form of matrix pair (K,M), ordered
% so that |TRIUA(j,j)/TRIUB(j,j) <= TRIUA(j+1, j+1)/TRIUB(j+1,j+1)|
[ TRIUA, TRIUB, Q, V ] = qz( K, M );
dist = GetLambda( diag(TRIUA), diag(TRIUB) );
dist = abs( dist - sigma );
[ ~ , idx ] = sort( dist, 'descend' );
clusters = idx;
for j = 1 : length( idx )
    clusters( idx(j) ) = j;
end
[ TRIUA, TRIUB, Q, V ] = ordqz( TRIUA, TRIUB, Q, V, clusters );
Q = Q';
end

function [ nlock, nact ] = GetNumLockAct( relres, tol )
% Determine the number of locked/active pairs
nev = length( relres );
nlock = 0;
for j = 1 : nev
    if ( relres( j ) < tol )
        nlock = nlock + 1;
    else break; end
end
nact = nev - nlock;
end

function [Q1, Q2, Q3] = SetupTestSubsp( AW, AS, AP, BW, BS, BP, Q, sigma )
% Setup test subspace (A - sigma*B)Z, where Z is a search subspace [V,W,S,P]
Q1 = AW - sigma*BW;
Q1 = Q1 - Q*( Q'*Q1 );
[ Q1, ~ ] = qr( Q1, 0 );

Q2 = AS - sigma*BS;
Q2 = Q2 - Q*( Q'*Q2 );
Q2 = Q2 - Q1*( Q1'*Q2 );
[ Q2, ~ ] = qr( Q2, 0 );

if ~isempty( AP )
    Q3 = AP - sigma*BP;
    Q3 = Q3 - Q*( Q'*Q3 );
    Q3 = Q3 - Q1*( Q1'*Q3 );
    Q3 = Q3 - Q2*( Q2'*Q3 );
    [ Q3, ~ ] = qr( Q3, 0 );
else
    Q3 = [];
end
end


function [ K, M ] = GetProjProb( AV, AW, AS, AP, BV, BW, BS, BP, Q, Q1, Q2, Q3 )
% Construct the projected (harmonic Rayleigh-Ritz) matrix pair
if ~isempty( AP )
    
    K = [ Q'*AV   Q'*AW   Q'*AS   Q'*AP
        Q1'*AV  Q1'*AW  Q1'*AS  Q1'*AP
        Q2'*AV  Q2'*AW  Q2'*AS  Q2'*AP
        Q3'*AV  Q3'*AW  Q3'*AS  Q3'*AP ];
    
    M = [ Q'*BV   Q'*BW   Q'*BS   Q'*BP
        Q1'*BV  Q1'*BW  Q1'*BS  Q1'*BP
        Q2'*BV  Q2'*BW  Q2'*BS  Q2'*BP
        Q3'*BV  Q3'*BW  Q3'*BS  Q3'*BP ];
else
    K = [ Q'*AV   Q'*AW   Q'*AS
        Q1'*AV  Q1'*AW  Q1'*AS
        Q2'*AV  Q2'*AW  Q2'*AS ];
    
    M = [ Q'*BV   Q'*BW   Q'*BS
        Q1'*BV  Q1'*BW  Q1'*BS
        Q2'*BV  Q2'*BW  Q2'*BS ];
end
end

function relres = GetRelres( W, lambda, nrmA, nrmB )
% Given block W of Schur residual vectors AV*MB - BV*MA ,
% compute relative norms for each resiudal
nev = size( W, 2 );
d = zeros( 1, nev );
for j = 1 : nev
    if ( abs(lambda(j)) ~= Inf )
        d( j ) = nrmA + abs(lambda( j ))*nrmB;	    
    else
        d( j ) = nrmB;	    
    end
end
relres = sqrt( abs(sum(conj(W).*W)) ) ./ d; 
end

function print_progress( msg, lambda, relres, nlock )
% Report current eigenvalue approximations in lambda
% and their associated relative residual norms in releres
fprintf('\n');
disp( msg );
nev = length( lambda );
for j = 1 : nev
    fprintf('Eigenvalue( %d ) = (%17.16e, %17.16e). RelRes = %4.3e \n', ...
        j, real( lambda(j) ), imag( lambda(j) ), relres( j ) );
end
fprintf('Number of converged eigenpairs = %d\n', nlock)
end

