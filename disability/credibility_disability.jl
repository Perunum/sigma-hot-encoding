# simulation example disability
include("pTUM-base-2.0.jl")
using Distributions
using Random
using Plots
using DataFrames
using CSV
using StatsPlots
using Measures

function disability_prob(v)
  disability_prob=v[2]*exp(v[1]*v[9])+v[5]*exp(v[4]*(v[9]-v[3])^2)+v[8]*exp(v[7]*(v[9]-v[6])^2)
end

function generate_data(n,ncats,age_min,age_max,cat_dist)
   
  # generate category covariate
  if cat_dist=="balanced"
    cat=(1).+floor.(rand(n)*ncats)
  elseif cat_dist=="skewed"
    cat=(1).+floor.(quantile.(Beta(2,3),rand(n))*ncats)
  else
  end

  # generate random effects categories
  α1=0.04.*ones(ncats).+(0.05-0.04).*rand(ncats)
  α2=0.0025.*ones(ncats).+(0.004-0.0025).*rand(ncats)
  β0=(30).*ones(ncats).+(40-30).*rand(ncats)
  β1=(-0.02).*ones(ncats).+(0-(-0.02)).*rand(ncats)
  β2=(0).*ones(ncats).+(0.0025-0).*rand(ncats)
  γ0=(65).*ones(ncats).+(70-65).*rand(ncats)
  γ1=(-0.02).*ones(ncats).+(0-(-0.02)).*rand(ncats)
  γ2=(-0.01).*ones(ncats).+(0-(-0.01)).*rand(ncats)
  δ= (-2).*ones(ncats).+(3-(-1)).*rand(ncats)
  u=[α1 α2 β0 β1 β2 γ0 γ1 γ2 δ]

  # generate data covariate and response
  x=zeros(n)
  y=zeros(n)
  for i=1:n
    v=u[round(Int,cat[i]),:]
    x[i]=round(age_min+(age_max-age_min)*quantile(Beta(max(2+v[9],2),max(2-v[9],2)),rand()))
    p=disability_prob([v[1:8]' x[i]])
    y[i]=(rand() < p ? 1 : 0)
  end

  data=[cat x y]
  return(u,data)
end

# settings
experiment_begin=2
experiment_end=2

run_begin=1
run_end=50

model_begin=1
model_end=3

n_train=500000
n_test=500000
ncats=18
layers=Int[4,3,2]
α=0.001 # learning rate
c=1.0 # maximum weight
iterations=15000

# collect stats
df_stats=DataFrame(experiment=Int64[],run=Int64[],model=Int64[],ME=Float64[],RMSE_avg=Float64[],LL=Float64[])

for experiment=experiment_begin:experiment_end

  # set parameters for experiment
  if experiment==1
    cat_dist="balanced"
    age_min=18
    age_max=67
  elseif experiment==2

    cat_dist="skewed"
    age_min=18
    age_max=67
  else
  end

  for run=run_begin:run_end
    Random.seed!(1234+experiment*100+run) # reproducable seed-value for each run
    u,data=generate_data(n_train+n_test,ncats,age_min,age_max,cat_dist)
    data_train=data[1:n_train,:]
    data_test=data[n_train+1:n_train+n_test,:]
    
    # compute mean and sd of variable age
    mean_x=mean(data_train[:,2])
    sd_x=std(data_train[:,2])

    # compress data_train and calculate weights
    df=DataFrame(cat=data_train[:,1],age=data_train[:,2],claim=data_train[:,3])
    df1=groupby(df,[:cat,:age,:claim])
    df1=combine(df1,[:cat] => ((c) -> w=sum(c)/maximum(c)) => :weight)
    data_train=Matrix{Float64}(df1)

    # compress data_test and calculate weights
    df=DataFrame(cat=data_test[:,1],age=data_test[:,2],claim=data_test[:,3])
    df1=groupby(df,[:cat,:age,:claim])
    df1=combine(df1,[:cat] => ((c) -> w=sum(c)/maximum(c)) => :weight)
    data_test=Matrix{Float64}(df1)

    # standardize x_train
    x_train=data_train[:,2]
    x_train_s=(x_train.-mean_x)./sd_x

    # standardize x_test
    x_test=data_test[:,2]
    x_test_s=(x_test.-mean_x)./sd_x

    for model=model_begin:model_end
      # 1: no categories in network
      # 2: 1-hot encoding (without weight balancing)
      # 3: σ-hot encoding (with weight balancing)

      y_train=data_train[:,3]
      weights_train=data_train[:,4]
      if model==1
        nvars=2 # x and y
        data_train_s=[x_train_s y_train weights_train]
      else
        hots=1.0*(1.0*collect(range(1,ncats)).==permutedims(data_train[:,1]))'
        if model==3
          hotvals=sqrt.(sum(hots.*weights_train,dims=1)/sum(weights_train).*((1).-sum(hots.*weights_train,dims=1)/sum(weights_train)))
        else  
          hotvals=ones(ncats)'
        end
        hots=hots.*hotvals
        nvars=ncats+2
        data_train_s=[hots x_train_s y_train weights_train]
      end

      w,b,npars=initCatDist(nvars,ncats,layers) # initialize weights and biases
      llv=Float64[] # log-likelihoods
      if model==3
        hotvars=[ncats]
      else
        hotvars=Int64[]
      end
      w,b,llv=fitCatDist(w,b,data_train_s,Float64[1,0],layers,iterations;hotvars=hotvars,obsweights=true,α=α,c=c)
      println("obs=",n_train,", parameters=",npars,", log-likelihood=",llv[length(llv)])

      y_test=data_test[:,3]
      weights_test=data_test[:,4]
      if model==1
        data_test_s=[x_test_s y_test weights_test]
      else
        hots_test=(1.0*(1.0*collect(range(1,ncats)).==permutedims(data_test[:,1]))').*hotvals
        data_test_s=[hots_test x_test_s y_test weights_test]
      end

      # predicted means
      y_pred=zeros(size(data_test,1))
      for i=1:size(data_test,1)
        y_pred[i]=evalCatPDF([data_test_s[i,1:nvars-1]' 1.0],Float64[1,0],layers,w,b)
      end

      # mean error (ME)
      ME=sum((y_pred.-y_test).*weights_test)/sum(weights_test)

      # log-likelihood and average mean absolute error per category (MAE_avg)
      LL=0
      y_test_c=zeros(ncats)
      y_pred_c=zeros(ncats)
      n_c=zeros(ncats)
      for i=1:size(data_test,1)
        cat=round(Int,data_test[i,1])
        y_test_c[cat]+=y_test[i]*weights_test[i]
        y_pred_c[cat]+=y_pred[i]*weights_test[i]
        LL+=log(evalCatPDF(data_test_s[i,1:nvars],Float64[1,0],layers,w,b))*weights_test[i]
        n_c[cat]+=weights_test[i]
      end
      n_c=max.(ones(ncats),n_c)
      RMSE_avg=sqrt(sum(((y_pred_c.-y_test_c)./n_c).^2)/ncats)
      LL=LL/sum(weights_test)

      println("experiment=",experiment)
      println("model=",model)
      println("run=",run)
      println("ME=",ME)
      println("RMSE_avg=",RMSE_avg)
      println("LL=",LL)
      push!(df_stats,[experiment run model ME RMSE_avg LL])
    end  
  end
end

# write and read stats
CSV.write("stats_disability_exp_2.csv",df_stats)

# boxplots of stats (RMSE_avg) for one experiment
df_1=CSV.read("stats_disability_exp_2.csv",DataFrame)
df_1.modelname = ifelse.(df_1.model .== 1," NN_nocats",ifelse.(df_1.model .== 2,"NN_1hot","NN_σhot")) # spaces in names for ordering boxes
p1=boxplot(df_1.modelname, df_1.RMSE_avg,label="RMSE_avg",bar_width=0.2,fillalpha=0.75,markeralpha=0.75,
          markercolor="blue",linecolor="black",fillcolor="blue")
plot(p1)
savefig(p1,"stats_disability_exp_2.pdf")
