# simulation friedman function, example Avanzi et al. 2023
include("pTUM-base-2.0.jl")
using Distributions
using Random
using Plots
using DataFrames
using CSV
using StatsPlots

function friedman(x)
  friedman=10*sin(π*x[1]*x[2])+20*(x[3]-0.5)^2+10*x[4]+5*x[5]
end

function generate_data(n,ncats,signal_to_noise,cat_dist,response_dist)
   
  # normalize signal_to_noise
  v=signal_to_noise./sum(signal_to_noise)
  μf,σu,σϵ=v[1],v[2],v[3]

  # generate random effects categories
  u=quantile.(Normal(0,σu),rand(ncats))

  # generate category covariate
  if cat_dist=="balanced"
    cat=(1).+floor.(rand(n)*ncats)
  elseif cat_dist=="skewed"
    cat=(1).+floor.(quantile.(Beta(2,3),rand(n))*ncats)
  else
  end

  # generate covariates x1..x10 and calculate scale parameter f
  x=zeros(n,10)
  t=0
  for i=1:n
    x[i,:]=rand(10)'
    t+=friedman(x[i,:])
  end
  scale_f=n*μf/t

  # generate response y
  y=zeros(n)
  for i=1:n
    v=scale_f*friedman(x[i,:])+u[round(Int,cat[i])]
    if response_dist=="gaussian-identity"
      y[i]=quantile(Normal(v,σϵ),rand())
    elseif response_dist=="gamma-exponential"
      y[i]=quantile(Gamma(1/σϵ^2,exp(v)*σϵ^2),rand())
    else
    end
  end
  data=[cat x y]
  return(μf,σu,σϵ,u,scale_f,data)
end

# settings
experiment_begin=6
experiment_end=6

run_begin=1
run_end=50

model_begin=1
model_end=3

n_train=5000
n_test=2500
ncats=100
layers1,layers2=Int[4,3],Int[2]
α=0.01 # learning rate
c=3.0 # maximum weight
iterations=15000

# collect stats
df_stats=DataFrame(experiment=Int64[],run=Int64[],model=Int64[],ME=Float64[],MAE=Float64[],RMSE=Float64[],RMSE_avg=Float64[],LL=Float64[])

for experiment=experiment_begin:experiment_end

  # set parameters for experiment
  if experiment==1
    signal_to_noise=[4,1,1]
    cat_dist="balanced"
    response_dist="gaussian-identity" # distribution - inverse link
  elseif experiment==2
    signal_to_noise=[4,1,1]
    cat_dist="balanced"
    response_dist="gamma-exponential" # distribution - inverse link
  elseif experiment==3
    signal_to_noise=[4,1,1]
    cat_dist="skewed"
    response_dist="gaussian-identity" # distribution - inverse link
  elseif experiment==4
    signal_to_noise=[4,1,2]
    cat_dist="balanced"
    response_dist="gaussian-identity" # distribution - inverse link
  elseif experiment==5
    signal_to_noise=[8,1,4]
    cat_dist="balanced"
    response_dist="gaussian-identity" # distribution - inverse link
  elseif experiment==6
    signal_to_noise=[8,1,4]
    cat_dist="skewed"
    response_dist="gamma-exponential" # distribution - inverse link
  else
  end

  for run=run_begin:run_end
    Random.seed!(1234+experiment*100+run) # reproducable seed-value for each run
    μf,σu,ϵ,u,scale_f,data=generate_data(n_train+n_test,ncats,signal_to_noise,cat_dist,response_dist)
    data_train=data[1:n_train,:]
    data_test=data[n_train+1:n_train+n_test,:]
    
    # standardize x_train
    x_train=data_train[:,2:11]
    mean_x=mean(x_train,dims=1)
    sd_x=std(x_train,dims=1)
    x_train_s=(x_train.-mean_x)./sd_x
    
    # standardize y_train
    y_train=data_train[:,12]
    mean_y=mean(y_train)
    sd_y=std(y_train)
    y_train_s=(y_train.-mean_y)./sd_y

    # standardize x_test
    x_test=data_test[:,2:11]
    x_test_s=(x_test.-mean_x)./sd_x

    # standardize y_test
    y_test=data_test[:,12]
    y_test_s=(y_test.-mean_y)./sd_y

    for model=model_begin:model_end
      # 1: no categories in network
      # 2: 1-hot encoding (without weight balancing)
      # 3: σ-hot encoding (with weight balancing)

      if model==1
        nvars=11 # x1..x10 and y
        data_train_s=[x_train_s y_train_s]
      else
        hots=1.0*(1.0*collect(range(1,ncats)).==permutedims(data_train[:,1]))'
        if model==3
          hotvals=sqrt.(sum(hots,dims=1)/n_train.*((1).-sum(hots,dims=1)/n_train))
        else  
          hotvals=ones(ncats)'
        end
        hots=hots.*hotvals
        nvars=ncats+11
        data_train_s=[hots x_train_s y_train_s]
      end
  
      w,b,npars=initDist(nvars,layers1,layers2) # initialize weights and biases
      llv=Float64[] # log-likelihoods
      if model==3
        hotvars=[ncats]
      else
        hotvars=Int64[]
      end
      w,b,llv=fitDist(w,b,data_train_s,layers1,layers2,iterations;hotvars=hotvars,α=α,c=c)
      println("obs=",n_train,", parameters=",npars,", log-likelihood=",llv[length(llv)])

      if model==1
        data_test_s=[x_test_s y_test_s]
      else
        hots_test=(1.0*(1.0*collect(range(1,ncats)).==permutedims(data_test[:,1]))').*hotvals
        data_test_s=[hots_test x_test_s y_test_s]
      end

      # loglikelihood and predicted means
      LL=0
      y_pred=zeros(n_test)
      for i=1:n_test
        LL+=log(evalPDF(data_test_s[i,:],layers1,layers2,w,b))
        for q=1:100
          y_pred[i]+=quantileCDF(data_test_s[i,1:nvars-1],(q-0.5)/100,layers1,layers2,w,b)/100
        end
      end
      y_pred=y_pred.*sd_y.+mean_y
      LL=LL/n_test
      
      # mean error (ME)
      ME=sum(y_pred.-y_test)/n_test

      # mean absolute error (MAE)
      MAE=sum(abs.(y_pred.-y_test))/n_test

      # root mean squared error (RMSE)
      RMSE=sqrt(sum((y_pred.-y_test).^2)/n_test)

      # average error of predicted mean per category (RMSE_avg)
      y_test_c=zeros(ncats)
      y_pred_c=zeros(ncats)
      n_c=zeros(ncats)
      for i=1:n_test
        cat=round(Int,data_test[i,1])
        y_test_c[cat]+=y_test[i]
        y_pred_c[cat]+=y_pred[i]
        n_c[cat]+=1
      end
      n_c=max.(ones(ncats),n_c)
      RMSE_avg=sqrt(sum(((y_pred_c.-y_test_c)./n_c).^2)/ncats)

      println("experiment=",experiment)
      println("model=",model)
      println("run=",run)
      println("ME=",ME)
      println("MAE=",MAE)
      println("RMSE=",RMSE)
      println("RMSE_avg=",RMSE_avg)
      println("LL=",LL)
      push!(df_stats,[experiment run model ME MAE RMSE RMSE_avg LL])
    end  
  end
end

# write and read stats
CSV.write("stats_friedman_exp_6.csv",df_stats)

# boxplots of stats (RMSE_avg) for one experiment
df_1=CSV.read("stats_friedman_exp_6.csv",DataFrame)
df_1.modelname = ifelse.(df_1.model .== 1," NN_nocats",ifelse.(df_1.model .== 2,"NN_1hot","NN_σhot")) # spaces in names for ordering boxes
p1=boxplot(df_1.modelname, df_1.RMSE_avg,label="RMSE_avg",bar_width=0.2,fillalpha=0.75,markeralpha=0.75,
          markercolor="blue",linecolor="black",fillcolor="blue")
df_2=CSV.read("stats_avanzi_2023_exp_6.csv",DataFrame) # Avanzi et al.,2023, downloads from github.com/agi-lab/glmmnet
df_2=filter([:model] => (x -> x == "GLMMNet" || x == "NN_ee" || x == "GPBoost"),df_2)
df_2.model = ifelse.(df_2.model .== "GLMMNet"," GLMMNet",ifelse.(df_2.model .== "NN_ee"," NN_ee","GPBoost")) # spaces in names for ordering boxes
p1=boxplot!(p1,df_2.model, df_2.RMSE_avg,label="RMSE_avg (Avanzi et al. 2023)",bar_width=0.2,fillalpha=0.75,markeralpha=0.75,
          markercolor="gray",linecolor="black",fillcolor="gray")
plot(p1)
savefig(p1,"stats_friedman_exp_6.pdf")

# plot (random example) categorical distributions
Random.seed!(1234)
cat=(1).+floor.(rand(n_train)*ncats)
countcats=sum(1.0*(1.0*collect(range(1,ncats)).==permutedims(cat)),dims=2)
p1=plot(collect(range(1,ncats)),countcats,st=:bar,color="green",ylims=(0,110),
legend=false,ylabel="count",title="balanced",fillalpha=0.75)
cat=(1).+floor.(quantile.(Beta(2,3),rand(n_train))*ncats)
countcats=sum(1.0*(1.0*collect(range(1,ncats)).==permutedims(cat)),dims=2)
p2=plot(collect(range(1,ncats)),countcats,st=:bar,color="green",ylims=(0,110),
legend=false,xlabel="category",ylabel="count",title="skewed",fillalpha=0.75)
p3=plot(p1,p2,layout=(2,1))
savefig(p3,"cat_distributions.pdf")
