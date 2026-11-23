import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


exp_types = ["NOE","JCOUPLINGS","CS","SAXS","RDC"]
bound_types = ["UPPER","LOWER"]
averaging_types = ["linear","power_3","power_6","power_4"]
fit_types = ["scale","scale+offset","no"]


# read a GROMACS .xvg file (e.g. the output of gmx gyrate)
def read_xvg(xvg_file,column=1):
    """Read one column of a GROMACS .xvg file.

    Lines starting with '#' or '@' are skipped; reading stops at the first '&'
    (end of the first data set). Column 0 is the time, column 1 the first data
    set (for gmx gyrate: the radius of gyration, in nm).

    Returns (time, values, legend), where legend is the label that GROMACS
    gives to the column (e.g. "Rg"), or None.
    """
    legends = {}
    rows = []
    ncols = None
    with open(xvg_file) as fh:
        for nline,line in enumerate(fh,1):
            s = line.strip()
            if not s or s[0]=="#":
                continue
            if s[0]=="@":
                # e.g.: @ s0 legend "Rg"
                fields = s[1:].split(None,2)
                if len(fields)==3 and fields[0].startswith("s") and fields[0][1:].isdigit() and fields[1]=="legend":
                    legends[int(fields[0][1:])+1] = fields[2].strip().strip('"')
                continue
            if s[0]=="&":
                break
            fields = s.split()
            if ncols is None:
                ncols = len(fields)
                if column<1 or column>=ncols:
                    raise ValueError("Column %d not available in %s: data lines have %d columns (0=time, 1..%d=data)" % (column,xvg_file,ncols,ncols-1))
            elif len(fields)!=ncols:
                raise ValueError("Line %d of %s has %d columns instead of %d" % (nline,xvg_file,len(fields),ncols))
            try:
                rows.append((float(fields[0]),float(fields[column])))
            except ValueError:
                raise ValueError("Cannot read numbers in line %d of %s" % (nline,xvg_file))
    if len(rows)==0:
        raise ValueError("No data found in %s" % xvg_file)
    data = np.array(rows)
    return data[:,0],data[:,1],legends.get(column)

# weighted average over samples: sum_k w_k calc_kj.
# A matrix-vector product, so that no temporary array of the size of calc is
# created (calc can be (1e6 frames x 1e2 data) for long trajectories)
def wavg(calc,sample_weights):
    return np.dot(np.asarray(sample_weights,dtype=float),calc)

# weighted variance of each column, computed in blocks of rows to limit memory
def _weighted_var(calc,sample_weights,mean,chunk=50000):
    var = np.zeros(calc.shape[1])
    for start in range(0,calc.shape[0],chunk):
        d = calc[start:start+chunk]-mean
        var += np.dot(sample_weights[start:start+chunk],d*d)
    return var/np.sum(sample_weights)

# calculate relative entropy
def srel(w0,w1):
    idxs = np.where(w1>1.0E-50)
    return np.sum(w1[idxs]*np.log(w1[idxs]/w0[idxs]))

# upper bound of the number of rows of a (possibly compressed) text file
def _count_lines(fname):
    from pandas.io.common import get_handle
    n = 0
    with get_handle(fname,"rb",compression="infer",is_text=False) as handles:
        fh = handles.handle
        last = b""
        while True:
            buf = fh.read(1<<24)
            if not buf:
                break
            n += buf.count(b"\n")
            last = buf[-1:]
    return n + (1 if last not in (b"",b"\n") else 0)

# read a calculated-data file (first column: frame number, not used) into a
# C-ordered float64 array (nsamples x ndata). Rows are counted first and the
# file is then read in blocks into a preallocated array, so that memory use
# stays close to the size of the data (files can be > 1 GB)
def read_calc(calc_file,chunksize=20000):
    kw = dict(sep=r"\s+",header=None,comment="#")
    ncols = pd.read_csv(calc_file,nrows=1,**kw).shape[1]
    if ncols<2:
        raise ValueError("Calculated data file %s must have at least two columns (frame, data...)" % calc_file)
    calc = np.empty((_count_lines(calc_file),ncols-1),dtype=np.float64)
    nrows = 0
    for chunk in pd.read_csv(calc_file,usecols=list(range(1,ncols)),dtype=np.float64,chunksize=chunksize,**kw):
        calc[nrows:nrows+chunk.shape[0]] = chunk.to_numpy(dtype=np.float64)
        nrows += chunk.shape[0]
    if nrows<calc.shape[0]:
        # comment or empty lines: keep only the rows that were read
        calc = calc[:nrows].copy() if nrows<0.99*calc.shape[0] else calc[:nrows]
    return calc

# parse files
def parse(exp_file,calc_file,averaging="auto"):

    log = ""
    # read experimental data 
    with open(exp_file) as fh:
        first = fh.readline()
    if not first.startswith("#") or "DATA=" not in first:
        raise ValueError("Error. First line of exp file %s must be in the format # DATA=[%s] [BOUND=UPPER/LOWER]" % (exp_file,exp_types))

    # read data type
    data_string = (first.split("DATA=")[-1].split()[0]).strip()
    if data_string not in exp_types:
        raise ValueError("Error. DATA in %s must be one of the following: %s " % (exp_file,exp_types))

    log += "# Reading %s data \n" % (data_string)
        
    # If it is not an average but a boundary it can be specified
    bound_string=None
    if(len(first.split("BOUND="))==2):
        bound_string = (first.split("BOUND=")[-1].split()[0]).strip()
        if bound_string not in bound_types:
            raise ValueError("Error. %s is not known. BOUND in %s must be one of the following: %s " % (bound_string,exp_file,bound_types))
        log += "# %s-bound data \n" % (bound_string)
        
    df_exp = pd.read_csv(exp_file,sep=r"\s+",header=None,comment="#")
        
    if df_exp.shape[1]!=3:
        raise ValueError("Error. Experimental datafile %s must be in the format LABEL VALUE ERROR" % exp_file)
    df_exp = df_exp.rename(columns={0: "label", 1: "val",2:"sigma"})
        
    # read calculated datafile
    calc = read_calc(calc_file)

    if (calc.shape[1])!=df_exp.shape[0]:
        raise ValueError("Error: Number of experimental data in %s (%d) must match the calculated data in %s (%d)" % (exp_file,df_exp.shape[0],calc_file,calc.shape[1]))

    # write to log file
    log += "# Reading %d experimental data from %s \n" % (df_exp.shape[0],exp_file)
    log += "# Reading %d calculated samples from %s \n" % (calc.shape[0],calc_file)
        
            
    # determine averaging
    if(averaging=="auto"):
        if(data_string=="NOE"):
            averaging = "power_6"
        else:
            averaging = "linear"
    else:
        if averaging not in averaging_types:
            raise ValueError("averaging type must be in %s " % (averaging_types))
    log += "# Using %s averaging \n" % (averaging)

    if(averaging.split("_")[0]=="power"):
        noe_power = int(averaging.split("_")[-1])
        df_exp["avg"] = np.power(df_exp["val"], -noe_power)
        df_exp["sigma2"] = (noe_power*df_exp["avg"]*df_exp["sigma"]/(df_exp["val"]))
        np.power(calc, -noe_power, out=calc)
            
        # if bound constraints, swap lower and upper
        if(bound_string=="LOWER"):
            bound_string="UPPER"
        elif(bound_string=="UPPER"):
            bound_string="LOWER"
    else:
        df_exp = df_exp.rename(columns={"val": "avg"})
        df_exp = df_exp.rename(columns={"sigma":"sigma2"})

    # define bounds 
    df_exp["bound"] = 0
    if(bound_string=="UPPER"): df_exp["bound"] = 1.0
    if(bound_string=="LOWER"): df_exp["bound"] = -1.0
    #df_exp["tag"] = exp_file

    labels = df_exp["label"].values
    exp = np.array(df_exp[["avg","sigma2","bound"]].values)
    return labels,exp,calc,log,averaging


# perform linear regression 
def fit_and_scale(exp,calc,sample_weights,fit):

    if fit not in fit_types:
        raise ValueError("fit type must be in %s " % (fit_types))

    calc_avg = wavg(calc,sample_weights)
    log = "# Using %s scaling \n" % (fit)
    if(fit=="no"):
        return calc_avg,log
    else:
        exp_avg = exp[:,0]

        if(fit=="scale"):
            fit_intercept=False
        else:
            fit_intercept=True

        oversigma = (1./exp[:,1]**2)

        reg = LinearRegression(fit_intercept=fit_intercept).fit(calc_avg.reshape(-1,1),exp_avg.reshape(-1,1),sample_weight=oversigma)
        r_value = reg.score(calc_avg.reshape(-1,1),exp_avg.reshape(-1,1),sample_weight=oversigma)
        # coef_ has shape (1,1) and intercept_ shape (1,) because y is 2D:
        # extract python floats (formatting 1-element arrays fails with numpy>=2)
        slope = float(np.ravel(reg.coef_)[0])
        intercept = float(np.ravel(reg.intercept_)[0])
               
        # calc is modified in place on purpose (callers use the scaled array)
        calc *=slope
        calc +=intercept

        calc_avg = wavg(calc,sample_weights)
        log += "# Slope=%8.4f; Offset=%8.4f; r2=%8.4f \n" % (slope,intercept,r_value)
        return calc_avg,log


# remove some of the samples
def subsample(label,exp,calc,use_samples,use_data):
    
    log = ""
    if(len(use_samples)!=0):
        calc = calc[use_samples,:]
        log += "# Using a subset of samples (%d) \n" % (calc.shape[0])
    if(len(use_data)!=0):
        label = label[use_data]
        exp = exp[use_data,:]
        calc = calc[:,use_data]
        log += "# Using a subset of datapoints (%d) \n" % (exp.shape[0])
    
    return label,exp, calc,log

# calculate chi2
def calc_chi(exp,calc,sample_weights):
    return calc_chi_avg(exp,wavg(calc,sample_weights))

# chi2 from the averages
def calc_chi_avg(exp,calc_avg):
    calc_avg = np.array(calc_avg,dtype=float)

    diff = (calc_avg-exp[:,0])    
    ii = np.where(((diff<0) & (exp[:,2]<0)) | ((diff>0) & (exp[:,2]>0)) )[0]
    ff = [1 if (exp[j,2]==0 or j in ii) else 0 for j in range(exp.shape[0])]
    diff *= ff #to_zero
    return  np.average((diff/exp[:,1])**2)

# sanity check 
def check_data(label,exp,calc,sample_weights):

    calc_avg = wavg(calc,sample_weights)

    log = ""

    diff = (calc_avg-exp[:,0])
    ii = np.where(((diff<0) & (exp[:,2]<0)) | ((diff>0) & (exp[:,2]>0)) )[0]
    ff = [1 if (exp[j,2]==0) else 0 for j in range(exp.shape[0])]
    nviol = 0
    if(len(ii)>0):
        log += "# The ensemble violates the following %d boundaries: \n" % (len(ii))
        log += "# %14s %8s %8s \n" % ("label","exp_avg","calc_avg")
        for j in ii:
            log  += "# %14s %8.4f %8.4f \n" % (label[j],exp[j,0],calc_avg[j])
            ff[j] = 1
            nviol += 1
    diff *= ff #to_zero
    chi2 = np.average((diff/exp[:,1])**2)
    rmsd = np.sqrt(np.average(diff**2))
    # data (not bounds) deviating by more than one sigma, in either direction
    ii_out =  np.where((np.abs(diff/exp[:,1])>1.) & (exp[:,2]==0))[0]
    nviol += len(ii_out)
    log += "CHI2: %.5f \n" % chi2
    log += "RMSD: %.5f \n" % rmsd
    log += "VIOLATIONS: %d \n" % nviol

    m_min = np.min(calc,axis=0)
    m_max = np.max(calc,axis=0)


    diff_min = ff*(m_min-exp[:,0])/exp[:,1]
    ii_min = np.where(diff_min>1.)[0]
    if(len(ii_min)>0):
        log += "##### WARNING ########## \n"
        log += "# The minimum value of the following data is higher than expt range: \n"
        log += "# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","min_calc")
        for j in ii_min:
            log += "# %15s %8.4f %8.4f %8.4f\n" % (label[j],exp[j,0],exp[j,1],m_min[j])

    diff_max = ff*(exp[:,0]-m_max)/exp[:,1]
    ii_max = np.where(diff_max>1.)[0]
    if(len(ii_max)>0):
        log += "##### WARNING ########## \n"
        log += "# The maximum value of the following data is lower than expt range: \n"
        log += "# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","max_calc")
        for j in ii_max:
            log += "# %15s %8.4f %8.4f %8.4f\n" % (label[j],exp[j,0],exp[j,1],m_max[j])
                          
    return log


# calculate Chi2, RMSD, number of violations
def calc_diffs(exp_avg,exp_sigma,bounds,calc_avg):

    diff = (calc_avg-exp_avg)/exp_sigma
    # boundaries violations
    ii = np.where(((diff<-1.) & (bounds<0)) | ((diff>1.) & (bounds>0)) )[0]
    ff = [1 if (bounds[j]==0 or j in ii) else 0 for j in range(exp_avg.shape[0])]
    # other violations
    viol = np.where(((diff**2>1) & (bounds==0)) )[0]
    #to_zero = np.zeros(len(diff))
    viol_idx = np.array(list(ii)+list(viol))
    nviol = viol_idx.shape[0]
    diff *= ff #to_zero
    chi2 = np.average((diff)**2)
    rmsd = np.sqrt(np.average((diff*exp_sigma)**2))
    return chi2,rmsd,nviol,viol_idx

# calculate statistics
def calc_stats(label,exp,calc,sample_weights0,sample_weights1,averaging,fit,outfile):
    
    if averaging not in averaging_types:
        raise ValueError("averaging type must be in %s " % (averaging_types))
    if fit not in fit_types:
        raise ValueError("fit type must be in %s " % (fit_types))
    if len(sample_weights1)==0:
        raise ValueError("Optimized weights are not available. Call fit() (or ibme()) before predict()")
    
    calc_avg0 = wavg(calc,sample_weights0)
    calc_avg1 = wavg(calc,sample_weights1)
    
    exp_avg = exp[:,0]
    oversigma = (1./exp[:,1]**2)
    #oversigma = np.ones(len(exp))
    intercept0,intercept1 = 0.,0.
    slope0,slope1 = 1.,1.
    
    if(fit=="scale"):
        slope0 = np.dot(exp_avg,calc_avg0)/np.dot(calc_avg0,calc_avg0)
        slope1 = np.dot(exp_avg,calc_avg1)/np.dot(calc_avg1,calc_avg1)        
    elif(fit=="scale+offset"):
        #slope0, intercept0, r_value0, p_value0, std_err0 = linregress(calc_avg0[:,0],exp[:,0])
        #slope1, intercept1, r_value1, p_value1, std_err1 = linregress(calc_avg1[:,0],exp[:,0])
        reg0 = LinearRegression(fit_intercept=True).fit(calc_avg0.reshape(-1,1),exp_avg.reshape(-1,1),sample_weight=oversigma)
        slope0,intercept0 = float(np.ravel(reg0.coef_)[0]),float(np.ravel(reg0.intercept_)[0])
        
        reg1 = LinearRegression(fit_intercept=True).fit(calc_avg1.reshape(-1,1),exp_avg.reshape(-1,1),sample_weight=oversigma)
        slope1,intercept1 = float(np.ravel(reg1.coef_)[0]),float(np.ravel(reg1.intercept_)[0])


    log = "# Slope=%8.4f/%8.4f; Offset=%8.4f/%8.4f \n" % (slope0,slope1,intercept0,intercept1)

    if(averaging.split("_")[0]=="power"):
        noe_power = int(averaging.split("_")[-1])

        calc_avg0 = np.power(slope0*calc_avg0+intercept0*np.sum(sample_weights0),-1/noe_power)
        calc_avg1 = np.power(slope1*calc_avg1+intercept1*np.sum(sample_weights1),-1/noe_power)
        exp_avg = np.power(exp[:,0],-1/noe_power)
        exp_sigma = (exp_avg*exp[:,1])/(noe_power*exp[:,0])
        bounds = -exp[:,2]
    else:
        calc_avg0 = slope0*calc_avg0+intercept0*np.sum(sample_weights0)
        calc_avg1 = slope1*calc_avg1+intercept1*np.sum(sample_weights1)

        exp_avg = exp[:,0]
        exp_sigma = exp[:,1]
        bounds = exp[:,2]
    log += "# %s averaging \n" % (averaging)
        
    
    chi2_0,rmsd_0,nviol_0, viol_idx_0 = calc_diffs(exp_avg,exp_sigma,bounds,calc_avg0)
    chi2_1,rmsd_1,nviol_1, viol_idx_1 = calc_diffs(exp_avg,exp_sigma,bounds,calc_avg1)

    if(outfile is not None):
        violation0 = [1 if k in viol_idx_0 else 0 for k in range(len(calc_avg0))]
        violation1 = [1 if k in viol_idx_1 else 0 for k in range(len(calc_avg1))] 
        violation = ["%d%d" % (i1,i2) for i1,i2 in zip(violation0,violation1)]
        
        df = pd.DataFrame({'label': label, 'exp_avg': exp_avg,'exp_sigma':exp_sigma,\
                           'calc_avg':calc_avg0,'calc_avg_rew':calc_avg1,"violation":violation})

        with open(outfile, 'w') as fh:
            fh.write('# %s \n' % " ".join(list(df.columns)))
        df['label'] = df['label'].map(lambda x: '%-15s' % x)
        df.to_csv(outfile,index=False,sep="\t",header=False,float_format="%8.3e",mode="a")
        with open(outfile, 'a') as fh:
            fh.write("# CHI2:       %8.4f %8.4f \n"  % (chi2_0,chi2_1))
            fh.write("# RMSD:       %8.4f %8.4f \n"  % (rmsd_0,rmsd_1))
            fh.write("# Violations: %8d %8d \n"  % (nviol_0,nviol_1))
            fh.write("# ndata:      %8d %8d \n"  % (exp.shape[0],calc.shape[0]))

    stats = [chi2_0,rmsd_0,nviol_0,chi2_1,rmsd_1,nviol_1]
    return stats,log


# parameters of the standardization x -> (x-v1)/v2, without modifying the data
def standardization(exp,calc,sample_weights,normalize="zscore"):
    ndata = exp.shape[0]
    if(normalize=="zscore"):
        v1 = wavg(calc,sample_weights)
        calc_var = _weighted_var(calc,sample_weights,v1)
        v2 = np.average(np.array([np.sqrt(calc_var),exp[:,1]]),axis=0)
        log = "# Z-score normalization \n"
    elif(normalize=="minmax"):
        v1 = calc.min(axis=0)
        v2 = calc.max(axis=0)-v1
        log = "# MinMax normalization \n"
    # do not use this one
    elif(normalize=="bme"):
        v1 = np.zeros(ndata)
        v2 = np.array(exp[:,0],dtype=float)
        log = "# BME normalization \n"
    elif(normalize=="none"):
        v1 = np.zeros(ndata)
        v2 = np.ones(ndata)
        log = "# No normalization \n"
    else:
        raise ValueError("normalize must be one of zscore, minmax, bme, none (got %s)" % normalize)
    return log,v1,v2

# standardize dataset. Modify array in place
def standardize(exp,calc,sample_weights,normalize="zscore"):

    log,v1,v2 = standardization(exp,calc,sample_weights,normalize=normalize)
    exp[:,0] -= v1
    exp[:,0] /= v2
    exp[:,1] /= v2
    calc -= v1
    calc /= v2
    if(normalize=="none"):
        v1,v2 = 0.,1.
    elif(normalize=="bme"):
        v1 = 0
    return log,v1,v2
