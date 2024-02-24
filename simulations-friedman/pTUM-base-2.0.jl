# pTUM-base v2.0.0.0

function initDist(nvars,layers1,layers2;shift=0.1)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  m1=max(nvars-(l1==0 ? 0 : 1),(l1>0 ? layers1[l1]+1 : 1))
  m2=max((l1>0 ? maximum(layers1) : 1),(l2>0 ? maximum(layers2) : 1))
  w=zeros(L,m2,max(m1,m2))
  b=zeros(L,m2)
  npars=0
  for l=1:L
    J=(l==1 ? nvars-((l1==0) ? 0 : 1) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : layers2[l-1-l1]))
    I=(l==L ? 1 : (l<=l1 ? layers1[l] : layers2[l-l1]))
    for i=1:I
      b[l,i]=atanh((2*i-I-1)/I)+shift
      npars+=J+1
    end
  end
  return(w,b,npars)
end

function fitDist(w,b,data,layers1,layers2,iterations;hotvars=Int64[],obsweights=false,α=0.001,β1=0.9,β2=0.999,ϵ=1e-8,c=3.0)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  N,nvars=size(data,1),size(data,2)-(obsweights ? 1 : 0)
  sw2,sw3,sb2=size(w,2),size(w,3),size(b,2)
  x,z,u,v,∂h∂x,∂h∂z,s1,s2=ntuple(_->zeros(L,sb2),8)
  llv=Float64[]
  # sum of observations per categorical value
  nhotvars=length(hotvars)
  if nhotvars>0
    hotsums=sum(data[:,1:sum(hotvars)] .* (obsweights ? data[:,nvars+1] : ones(N)),dims=1)
    hotvals=minimum(data[:,1:sum(hotvars)],dims=1)+maximum(data[:,1:sum(hotvars)],dims=1)
    nobs= (obsweights ? sum(data[:,nvars+1]) : N)
  end
  # Adam optimization
  mb,vb=ntuple(_->zeros(L,sb2),2)
  mw,vw=ntuple(_->zeros(L,sw2,sw3),2)
  println()
  println("Maximizing likelihood continuous distribution..")
  for iteration=0:iterations
    ∂h∂w,∂h∂b=zeros(L,sw2,sw3),zeros(L,sb2)
    h=0
    for n=1:N
      obs=data[n,:]
      obsw=(obsweights ? obs[nvars+1] : 1)
      # propagate forward
      for l=1:L
        J = (l==1 ? nvars-((l1==0) ? 0 : 1) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : layers2[l-1-l1]))
        I = (l==L ? 1 : (l<=l1 ? layers1[l] : layers2[l-l1]))
        for i=1:I
          u[l,i]=b[l,i]
          v[l,i]=0
          for j=1:J
            u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : (((l==l1+1)&&(j==J)) ? obs[nvars] : x[l-1,j]))
            if l>=l1+1
              v[l,i]+=w[l,i,j]*(l==l1+1 ? (j==J ? 1 : 0) : z[l-1,j])
            end
          end
          x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
          s1[l,i]=(l==L ? x[l,i]*(1-x[l,i]) : 1-x[l,i]^2)
          s2[l,i]=(l==L ? (1-2*x[l,i])*s1[l,i] : -2*x[l,i]*s1[l,i])
          z[l,i]=(l<=l1 ? 0 : max(s1[l,i]*v[l,i],1e-9))
        end
      end
      h+=obsw*log(z[L,1])
      ∂h∂x[L,1]=0
      ∂h∂z[L,1]=obsw/z[L,1]
      # propagate backward
      if iteration<iterations
        l=L-1
        while l>=0
          I=(l==0 ? nvars-((l1==0) ? 0 : 1) : (l<=l1 ? (layers1[l]+(l==l1 ? 1 : 0)) : layers2[l-l1]))
          J=(l+1==L ? 1 : (l+1<=l1 ? layers1[l+1] : layers2[l+1-l1]))
          for i=1:I
            if (l>0)&&((i<I)||(l!=l1))
              ∂h∂x[l,i]=0
              ∂h∂z[l,i]=0
            end
            for j=1:J
              if (l>0)&&((i<I)||(l!=l1))
                ∂h∂x[l,i]+=w[l+1,j,i]*(∂h∂x[l+1,j]*s1[l+1,j]+∂h∂z[l+1,j]*s2[l+1,j]*v[l+1,j])
                ∂h∂z[l,i]+=w[l+1,j,i]*∂h∂z[l+1,j]*s1[l+1,j]
              end
              if i==1
                ∂h∂b[l+1,j]+=∂h∂x[l+1,j]*s1[l+1,j]+∂h∂z[l+1,j]*s2[l+1,j]*v[l+1,j]
                if n==N
                  mb[l+1,j]=β1*mb[l+1,j]+(1-β1)*∂h∂b[l+1,j]
                  vb[l+1,j]=β2*vb[l+1,j]+(1-β2)*∂h∂b[l+1,j]^2
                  b[l+1,j]+=α*mb[l+1,j]/((1-β1^(iteration+1))*(sqrt(vb[l+1,j]/(1-β2^(iteration+1)))+ϵ))
                end
              end
              ∂h∂w[l+1,j,i]+=∂h∂x[l+1,j]*s1[l+1,j]*(l==0 ? obs[i] : (((l==l1)&&(i==I)) ? obs[nvars] : x[l,i]))+∂h∂z[l+1,j]*
              (s2[l+1,j]*v[l+1,j]*(l==0 ? obs[i] : (((l==l1)&&(i==I)) ? obs[nvars] : x[l,i]))+s1[l+1,j]*(((l==l1)&&(i==I)) ? (i==I ? 1 : 0) : (l==0 ? 0 : z[l,i])))
              if n==N
                mw[l+1,j,i]=β1*mw[l+1,j,i]+(1-β1)*∂h∂w[l+1,j,i]
                vw[l+1,j,i]=β2*vw[l+1,j,i]+(1-β2)*∂h∂w[l+1,j,i]^2
                w[l+1,j,i]=max(-c,min(c,w[l+1,j,i]+α*mw[l+1,j,i]/((1-β1^(iteration+1))*(sqrt(vw[l+1,j,i]/(1-β2^(iteration+1)))+ϵ))))
                if l>l1 || ((l==l1)&&(i==I))
                  w[l+1,j,i]=max(w[l+1,j,i],0)
                end
              end
            end
          end
          l-=1
        end
      end
    end
    # balance hot-encoded categorical variables
    if (nhotvars>0)&&(iteration<iterations)
      J=(L==1 ? 1 : (l1>=1 ? layers1[1] : layers2[1]))
      for j=1:J
        i=1
        for hotvar=1:nhotvars
          inbalance=0
          hots=hotvars[hotvar]
          if hots==1
            i+=1
            break
          end
          ii=hots
          while ii>0
            inbalance+=w[1,j,ii]*hotsums[ii]
            ii-=1
            i+=1
          end
          inbalance=inbalance/nobs
          b[1,j]+=inbalance
          for ii=1:hots
            w[1,j,i-ii]=max(-c,min(c,w[1,j,i-ii]-(hotvals[i-ii]==0 ? 0 : inbalance/hotvals[i-ii])))
          end
        end
      end
    end
    push!(llv,h)
    if (mod(iteration,100)==0)||(iteration==iterations)  
      print("\e[2K")
      print("\e[1G")
      print("iteration ",iteration,"/",iterations,", log-likelihood=",h)
    end
  end
  println()
  return(w,b,llv)
end

function evalPDF(obs,layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  nvars=length(obs)
  x,z,u,v,s1=ntuple(_->zeros(L,size(b,2)),5)
  for l=1:L
    J = (l==1 ? nvars-((l1==0) ? 0 : 1) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : layers2[l-1-l1]))
    I = (l==L ? 1 : (l<=l1 ? layers1[l] : layers2[l-l1]))
    for i=1:I
      u[l,i]=b[l,i]
      v[l,i]=0
      for j=1:J
        u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : (((l==l1+1)&&(j==J)) ? obs[nvars] : x[l-1,j]))
        if l>=l1+1
          v[l,i]+=w[l,i,j]*(l==l1+1 ? (j==J ? 1 : 0) : z[l-1,j])
        end
      end
      x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
      s1[l,i]=(l==L ? x[l,i]*(1-x[l,i]) : 1-x[l,i]^2)
      z[l,i]=(l<=l1 ? 0 : s1[l,i]*v[l,i])
    end
  end
  return(z[L,1])
end

function evalCDF(obs,layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  nvars=length(obs)
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  for l=1:L
    J = (l==1 ? nvars-((l1==0) ? 0 : 1) : (l<=l1+1 ? (layers1[l-1]+(l==l1+1 ? 1 : 0)) : layers2[l-1-l1]))
    I = (l==L ? 1 : (l<=l1 ? layers1[l] : layers2[l-l1]))
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : (((l==l1+1)&&(j==J)) ? obs[nvars] : x[l-1,j]))
      end
      x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
    end
  end
  return(x[L,1])
end

function limitsCDF(layers1,layers2,w,b)
  l1,l2=length(layers1),length(layers2)
  L=l1+l2+1
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  # calculate infimum
  for l=(l1+1):L
    J = (l==l1+1 ? 1 : layers2[l-1-l1])
    I = (l==L ? 1 : layers2[l-l1])
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*((l==l1+1) ? 0 : x[l-1,j])
      end
      x[l,i]=(l==L ? ((l==l1+1) ? 0 : 1/(1+exp(-u[l,i]))) : ((l==l1+1) ? -1 : tanh(u[l,i])))
    end
  end
  inf=x[L,1]
  # calculate supremum
  for l=(l1+1):L
    J = (l==l1+1 ? 1 : layers2[l-1-l1])
    I = (l==L ? 1 : layers2[l-l1])
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*((l==l1+1) ? 0 : x[l-1,j])
      end
      x[l,i]=(l==L ? ((l==l1+1) ? 1 : 1/(1+exp(-u[l,i]))) : ((l==l1+1) ? 1 : tanh(u[l,i])))
    end
  end
  sup=x[L,1]
  return(inf,sup)
end

function quantileCDF(cvars,p,layers1,layers2,w,b;accuracy=1e-8,boundLow=-1e6,boundUpp=1e6)
  inf,sup=limitsCDF(layers1,layers2,w,b)
  u=inf+(sup-inf)*p
  xLow,xUpp=-1,1
  yLow=evalCDF([cvars;xLow],layers1,layers2,w,b)
  yUpp=evalCDF([cvars;xUpp],layers1,layers2,w,b)
  while ((yLow>u)&&(xLow>boundLow))
    xUpp,yUpp=xLow,yLow
    xLow=2*xLow
    yLow=evalCDF([cvars;xLow],layers1,layers2,w,b)
  end
  while ((yUpp<u)&&(xUpp<boundUpp))
    xLow,yLow=xUpp,yUpp
    xUpp=2*xUpp
    yUpp=evalCDF([cvars;xUpp],layers1,layers2,w,b)
  end
  while ((xUpp-xLow)>accuracy)
    xInt=xLow+(xUpp-xLow)/2
    yInt=evalCDF([cvars;xInt],layers1,layers2,w,b)
    if (yInt>u)
      xUpp,yUpp=xInt,yInt
    else
      xLow,yLow=xInt,yInt
    end
  end
  return(xLow+(xUpp-xLow)/2)
end

function initCatDist(nvars,ncats,layers;shift=0.1)
  L=length(layers)+1
  maxWidth=(length(layers)==0 ? 0 : maximum(layers))
  w=zeros(L,max(ncats-1,maxWidth),max(nvars-1,maxWidth))
  b=zeros(L,max(ncats-1,maxWidth))
  npars=0
  for l=1:L
    J=(l==1 ? nvars-1 : layers[l-1])
    I=(l==L ? ncats-1 : layers[l])
    for i=1:I
      b[l,i]=atanh((2*i-I-1)/I)+shift
      npars+=J+1
    end
  end
  return(w,b,npars)
end

function fitCatDist(w,b,data,categories,layers,iterations;hotvars=Int64[],obsweights=false,α=0.001,β1=0.9,β2=0.999,ϵ=1e-8,c=3.0)
  L=length(layers)+1
  ncats=length(categories)
  N,nvars=size(data,1),size(data,2)-(obsweights ? 1 : 0)
  sw2,sw3,sb2=size(w,2),size(w,3),size(b,2)
  x,u,∂h∂x,s1=ntuple(_->zeros(L,sb2),4)
  llv=Float64[]
  # sum of observations per categorical value
  nhotvars=length(hotvars)
  if nhotvars>0
    hotsums=sum(data[:,1:sum(hotvars)] .* (obsweights ? data[:,nvars+1] : ones(N)),dims=1)
    hotvals=minimum(data[:,1:sum(hotvars)],dims=1)+maximum(data[:,1:sum(hotvars)],dims=1)
    nobs= (obsweights ? sum(data[:,nvars+1]) : N)
  end
  # Adam optimization
  mb,vb=ntuple(_->zeros(L,sb2),2)
  mw,vw=ntuple(_->zeros(L,sw2,sw3),2)
  println()
  println("Maximizing likelihood categorical distribution..")
  for iteration=0:iterations
    ∂h∂w,∂h∂b=zeros(L,sw2,sw3),zeros(L,sb2)
    h=0
    for n=1:N
      obs=data[n,:]
      obsw=(obsweights ? obs[nvars+1] : 1)
      catnr=ncats
      # propagate forward
      for l=1:L
        J = (l==1 ? nvars-1 : layers[l-1])
        I = (l==L ? ncats-1 : layers[l])
        for i=1:I
          u[l,i]=b[l,i]
          for j=1:J
            u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : x[l-1,j])
          end
          x[l,i]=(l==L ? 1/(1+exp(-u[l,i])) : tanh(u[l,i]))
          s1[l,i]=(l==L ? x[l,i]*(1-x[l,i]) : 1-x[l,i]^2)
          if (l==L)
            if obs[nvars]==categories[i]
              catnr=i
            end
            ∂h∂x[l,i]=obsw*(i<catnr ? 1/(x[l,i]-1) : (i==catnr ? 1/x[l,i] : 0))
            h+=obsw*(i<catnr ? log(1-x[l,i]) : (i==catnr ? log(x[l,i]) : 0))
          end
        end
      end
      # propagate backward
      if iteration<iterations
        l=L-1
        while l>=0
          I=(l==0 ? nvars-1 : layers[l])
          J=(l==L-1 ? ncats-1 : layers[l+1])
          for i=1:I
            if (l>0)
              ∂h∂x[l,i]=0
            end
            for j=1:J
              if (l>0)
                ∂h∂x[l,i]+=w[l+1,j,i]*∂h∂x[l+1,j]*s1[l+1,j]
              end
              if i==1
                ∂h∂b[l+1,j]+=∂h∂x[l+1,j]*s1[l+1,j]
                if n==N
                  mb[l+1,j]=β1*mb[l+1,j]+(1-β1)*∂h∂b[l+1,j]
                  vb[l+1,j]=β2*vb[l+1,j]+(1-β2)*∂h∂b[l+1,j]^2
                  b[l+1,j]+=α*mb[l+1,j]/((1-β1^(iteration+1))*(sqrt(vb[l+1,j]/(1-β2^(iteration+1)))+ϵ))
                end
              end
              ∂h∂w[l+1,j,i]+=∂h∂x[l+1,j]*s1[l+1,j]*(l==0 ? obs[i] : x[l,i])
              if n==N
                mw[l+1,j,i]=β1*mw[l+1,j,i]+(1-β1)*∂h∂w[l+1,j,i]
                vw[l+1,j,i]=β2*vw[l+1,j,i]+(1-β2)*∂h∂w[l+1,j,i]^2
                w[l+1,j,i]=max(-c,min(c,w[l+1,j,i]+α*mw[l+1,j,i]/((1-β1^(iteration+1))*(sqrt(vw[l+1,j,i]/(1-β2^(iteration+1)))+ϵ))))
              end
            end
          end
          l-=1
        end
      end
    end
    # balance hot-encoded categorical variables
    if (nhotvars>0)&&(iteration<iterations)
      J=(L==1 ? ncats-1 : layers[1])
      for j=1:J
        i=1
        for hotvar=1:nhotvars
          inbalance=0
          hots=hotvars[hotvar]
          if hots==1
            i+=1
            break
          end
          ii=hots
          while ii>0
            inbalance+=w[1,j,ii]*hotsums[ii]
            ii-=1
            i+=1
          end
          inbalance=inbalance/nobs
          b[1,j]+=inbalance
          for ii=1:hots
            w[1,j,i-ii]=max(-c,min(c,w[1,j,i-ii]-(hotvals[i-ii]==0 ? 0 : inbalance/hotvals[i-ii])))
          end
        end
      end
    end
    push!(llv,h)
    if (mod(iteration,100)==0)||(iteration==iterations)  
      print("\e[2K")
      print("\e[1G")
      print("iteration ",iteration,"/",iterations,", log-likelihood=",h)
    end
  end
  println()
  return(w,b,llv)
end

function evalCatPDF(obs,categories,layers,w,b)
  L=length(layers)+1
  nvars=length(obs)
  ncats=length(categories)
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  p=1
  for l=1:L
    J=(l==1 ? nvars-1 : layers[l-1])
    I=(l==L ? ncats-1 : layers[l])
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : x[l-1,j])
      end
      x[l,i]=(l==L ? p/(1+exp(-u[l,i])) : tanh(u[l,i]))
      if (l==L)&&(obs[nvars]==categories[i])
        p=x[L,i]
        break
      end
      p-=(l==L ? x[l,i] : 0)
    end
  end
  return(p)
end

function evalCatCDF(obs,categories,layers,w,b)
  L=length(layers)+1
  nvars=length(obs)
  ncats=length(categories)
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  p=1
  for l=1:L
    J=(l==1 ? nvars-1 : layers[l-1])
    I=(l==L ? ncats-1 : layers[l])
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*(l==1 ? obs[j] : x[l-1,j])
      end
      x[l,i]=(l==L ? 1-(1-1/(1+exp(-u[l,i])))*(1-(i==1 ? 0 : x[l,i-1])) : tanh(u[l,i]))
      if (l==L)&&(obs[nvars]==categories[i])
        p=x[L,i]
        break
      end
    end
  end
  return(p)
end

function quantileCatCDF(cvars,p,categories,layers,w,b)
  L=length(layers)+1
  nvars=length(cvars)+1
  ncats=length(categories)
  x,u=ntuple(_->zeros(L,size(b,2)),2)
  catnr=ncats
  for l=1:L
    J=(l==1 ? nvars-1 : layers[l-1])
    I=(l==L ? ncats-1 : layers[l])
    for i=1:I
      u[l,i]=b[l,i]
      for j=1:J
        u[l,i]+=w[l,i,j]*(l==1 ? cvars[j] : x[l-1,j])
      end
      x[l,i]=(l==L ? 1-(1-1/(1+exp(-u[l,i])))*(1-(i==1 ? 0 : x[l,i-1])) : tanh(u[l,i]))
      if (l==L)&&(x[l,i]>=p)
        catnr=i
        break
      end
    end
  end
  return(categories[catnr])
end
