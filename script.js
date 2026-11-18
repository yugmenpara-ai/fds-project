'use strict';

/* FDS Music Lab — dependency-free statistical engine, for static site hosting. */
const FDS = (() => {
  const ALPHA = 0.05;
  const CATEGORIES = ['Less than 30 minutes','30-60 minutes','1-2 hours','2-3 hours','More than 3 hours'];
  const LABELS = ['<30 min','30–60 min','1–2 h','2–3 h','>3 h'];
  const MIDPOINTS = [0.25, 0.75, 1.5, 2.5, 3.5];
  const REFERENCE_COUNTS = [38,61,52,42,18];
  const ALIASES = new Map([
    ['less than 30 minutes',0],['less than 30 min',0],['<30 min',0],['<30 minutes',0],
    ['30-60 minutes',1],['30-60 min',1],['1-2 hours',2],['1-2h',2],['1-2 h',2],
    ['2-3 hours',3],['2-3h',3],['2-3 h',3],['more than 3 hours',4],['more than 3h',4],
    ['>3 h',4],['>3h',4]
  ]);
  function normalizeCategory(raw) {
    const text = String(raw ?? '').trim().toLowerCase().replace(/[–—−]/g,'-').replace(/\s+/g,' ');
    if (!ALIASES.has(text)) throw new Error(`Unknown music_time category: ${JSON.stringify(raw)}`);
    return ALIASES.get(text);
  }
  function validateCounts(counts) {
    if (!Array.isArray(counts) || counts.length !== 5 || counts.some(x=>!Number.isSafeInteger(x) || x<0)) throw new Error('Enter five non-negative whole-number counts.');
    const n = counts.reduce((a,b)=>a+b,0);
    if (n<2) throw new Error('At least two new student responses are required.');
    if (n>10000) throw new Error('This prototype accepts at most 10,000 new responses.');
    return n;
  }
  function stats(counts) {
    const n=counts.reduce((a,b)=>a+b,0);
    if (n<2) throw new Error('At least two responses are required.');
    const mean=counts.reduce((s,c,i)=>s+c*MIDPOINTS[i],0)/n;
    const variance=counts.reduce((s,c,i)=>s+c*(MIDPOINTS[i]-mean)**2,0)/(n-1);
    return {n,mean,sd:Math.sqrt(Math.max(variance,0)),variance,counts:counts.slice()};
  }
  // Lanczos approximation for log Γ(z), including its reflection branch.
  function logGamma(z){
    const p=[0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];
    if(z<0.5)return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*z))-logGamma(1-z);
    z-=1;let x=p[0];for(let i=1;i<p.length;i++)x+=p[i]/(z+i);
    const t=z+7.5;return 0.5*Math.log(2*Math.PI)+(z+0.5)*Math.log(t)-t+Math.log(x);
  }
  // Numerical Recipes continued fraction for the incomplete beta function.
  function betaCF(a,b,x){
    const MAX=300, EPS=3e-14, TINY=1e-300;
    let qab=a+b,qap=a+1,qam=a-1,c=1,d=1-qab*x/qap;
    if(Math.abs(d)<TINY)d=TINY;d=1/d;let h=d;
    for(let m=1;m<=MAX;m++){
      let m2=2*m;let aa=m*(b-m)*x/((qam+m2)*(a+m2));
      d=1+aa*d;if(Math.abs(d)<TINY)d=TINY;
      c=1+aa/c;if(Math.abs(c)<TINY)c=TINY;d=1/d;h*=d*c;
      aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
      d=1+aa*d;if(Math.abs(d)<TINY)d=TINY;
      c=1+aa/c;if(Math.abs(c)<TINY)c=TINY;d=1/d;
      const delta=d*c;h*=delta;if(Math.abs(delta-1)<EPS)break;
    }
    return h;
  }
  function incBeta(a,b,x){
    if(x<=0)return 0;if(x>=1)return 1;
    const bt=Math.exp(logGamma(a+b)-logGamma(a)-logGamma(b)+a*Math.log(x)+b*Math.log1p(-x));
    const val=x<(a+1)/(a+b+2)?bt*betaCF(a,b,x)/a:1-bt*betaCF(b,a,1-x)/b;
    return Math.min(1,Math.max(0,val));
  }
  function twoSidedTP(t,df){const x=df/(df+t*t);return incBeta(df/2,0.5,x);}
  function tCritical(df,alpha=ALPHA){
    let lo=0,hi=1;while(twoSidedTP(hi,df)>alpha && hi<1e7)hi*=2;
    for(let i=0;i<90;i++){const mid=(lo+hi)/2;if(twoSidedTP(mid,df)>alpha)lo=mid;else hi=mid;}
    return (lo+hi)/2;
  }
  function chiSquareSF(stat,df){
    // Q(a,x), regularized upper incomplete gamma.
    const a=df/2,x=stat/2,EPS=3e-14,MIN=1e-300;
    if(x<=0)return 1;
    const gammaTerm=a*Math.log(x)-x-logGamma(a);
    if(x<a+1){let sum=1/a,term=sum,ap=a;for(let i=1;i<=500;i++){ap++;term*=x/ap;sum+=term;if(Math.abs(term)<Math.abs(sum)*EPS)break;}return Math.max(0,Math.min(1,1-sum*Math.exp(gammaTerm)));}
    let b=x+1-a,c=1/MIN,d=1/Math.max(b,MIN),h=d;
    for(let i=1;i<=500;i++){const an=-i*(i-a);b+=2;d=an*d+b;if(Math.abs(d)<MIN)d=MIN;c=b+an/c;if(Math.abs(c)<MIN)c=MIN;d=1/d;const delta=d*c;h*=delta;if(Math.abs(delta-1)<EPS)break;}
    return Math.min(1,Math.max(0,Math.exp(gammaTerm)*h));
  }
  function welchTest(counts){
    const newest=stats(counts),reference=stats(REFERENCE_COUNTS);
    const vn=newest.variance/newest.n, vr=reference.variance/reference.n;
    const se=Math.sqrt(vn+vr);
    if(!se)throw new Error('Both samples have zero variation; Welch t-test is undefined.');
    const df=(vn+vr)**2/(vn**2/(newest.n-1)+vr**2/(reference.n-1));
    const difference=newest.mean-reference.mean,t=difference/se,p=twoSidedTP(t,df),critical=tCritical(df);
    return {reference,newest,difference,t,df,p,critical,ciLow:difference-critical*se,ciHigh:difference+critical*se,se,reject:p<ALPHA};
  }
  function seededRandom(seed){let x=seed|0;return ()=>{x^=x<<13;x^=x>>>17;x^=x<<5;return (x>>>0)/4294967296;};}
  function distributionTest(counts,simulations=10000){
    const ref=REFERENCE_COUNTS, nRef=ref.reduce((a,b)=>a+b),nNew=counts.reduce((a,b)=>a+b),n=nRef+nNew;
    const colTotals=ref.map((v,i)=>v+counts[i]);
    const expectedRef=colTotals.map(v=>v*nRef/n),expectedNew=colTotals.map(v=>v*nNew/n);
    const score=(rowRef)=>rowRef.reduce((s,v,i)=>{
      const other=colTotals[i]-v;
      return s+(v-expectedRef[i])**2/expectedRef[i]+(other-expectedNew[i])**2/expectedNew[i];
    },0);
    const statistic=score(ref),df=4,minExpected=Math.min(...expectedRef,...expectedNew);
    if(minExpected>=5)return {statistic,df,minExpected,p:chiSquareSF(statistic,df),method:'Pearson χ² · asymptotic p-value',monteCarlo:false};
    // Fixed-margin 2x5 random tables via sequential hypergeometric draws.
    // Row and column margins are held constant for every simulation.
    const random=seededRandom(2026),rows=simulations;
    let extreme=0;
    for(let k=0;k<rows;k++){
      let rowRemaining=nRef,popRemaining=n;const drawn=[];
      for(let i=0;i<5;i++){
        if(i===4){drawn.push(rowRemaining);break;}
        let successesLeft=colTotals[i],inGroup=0;
        for(let j=0;j<rowRemaining;j++){
          if(random()<successesLeft/(popRemaining-j)){inGroup++;successesLeft--;}
        }
        drawn.push(inGroup);rowRemaining-=inGroup;popRemaining-=colTotals[i];
      }
      if(score(drawn)>=statistic-1e-10)extreme++;
    }
    return {statistic,df,minExpected,p:(extreme+1)/(rows+1),method:`Fixed-margin Monte Carlo · ${rows.toLocaleString('en-US')} draws`,monteCarlo:true};
  }
  function parseCsv(text){
    // RFC 4180-style quoting; handles embedded commas, quoted newlines, CRLF, UTF-8 BOM.
    text=text.replace(/^\uFEFF/,'');let rows=[],row=[],field='',inside=false,closed=false,rowHadQuotes=false;
    for(let i=0;i<text.length;i++){
      const ch=text[i];
      if(ch==='"'){
        if(inside && text[i+1]==='"'){field+='"';i++;}
        else if(inside){inside=false;closed=true;}
        else if(field===''){inside=true;rowHadQuotes=true;}
        else throw new Error('Malformed CSV: unexpected quote.');
      } else if(ch===',' && !inside){row.push(field);field='';closed=false;}
      else if((ch==='\n'||ch==='\r')&&!inside){if(ch==='\r'&&text[i+1]==='\n')i++;row.push(field);if(rowHadQuotes||row.some(v=>v.trim()!==''))rows.push(row);row=[];field='';closed=false;rowHadQuotes=false;}
      else {if(closed&&!/\s/.test(ch))throw new Error('Malformed CSV: characters after a closing quote.');field+=ch;}
    }
    if(inside)throw new Error('Malformed CSV: unclosed quote.');
    row.push(field);if(rowHadQuotes||row.some(v=>v.trim()!==''))rows.push(row);
    if(rows.length===0)throw new Error('The CSV is empty.');
    const headers=rows.shift().map(s=>s.trim());const idx=headers.indexOf('music_time');
    if(idx<0)throw new Error('CSV needs a column named exactly music_time.');
    if(rows.length===0)throw new Error('The CSV contains no student responses.');
    if(rows.length>10000)throw new Error('The CSV exceeds the 10,000-response limit.');
    const counts=[0,0,0,0,0];
    for(let i=0;i<rows.length;i++){
      const raw=rows[i][idx];
      if(raw===undefined||raw.trim()==='')throw new Error(`Row ${i+2}: missing music_time value.`);
      let bucket;
      try{bucket=normalizeCategory(raw);}catch(e){throw new Error(`Row ${i+2}: ${e.message}`);}
      counts[bucket]++;
    }
    return counts;
  }
  function test(counts){validateCounts(counts);return {counts:counts.slice(),welch:welchTest(counts),chi:distributionTest(counts)};}
  return {ALPHA,CATEGORIES,LABELS,MIDPOINTS,REFERENCE_COUNTS,normalizeCategory,validateCounts,stats,welchTest,distributionTest,parseCsv,test,twoSidedTP,tCritical,chiSquareSF,logGamma};
})();

// Browser interface. Only activate when the page contains the app elements.
if(typeof document!=='undefined' && document.getElementById('runBtn')){
  const $=(id)=>document.getElementById(id);
  let mode='counts',uploadedCounts=null,uploadedFileName='',lastReport=null;
  const countInputs=Array.from({length:5},(_,i)=>$('count'+i));
  function fmt(v,d=3){return Number.isFinite(v)?v.toFixed(d):'—';}
  function pFmt(p){if(p<0.0001)return p<1e-15?'< 1 × 10⁻¹⁵':p.toExponential(2);return p.toFixed(4);}
  function notify(msg,good=false){$('message').textContent=msg;$('message').className='message'+(good?' good':'');$('message').hidden=false;}
  function clearMessage(){$('message').hidden=true;}
  function switchMode(next){mode=next;$('tabCounts').classList.toggle('selected',next==='counts');$('tabUpload').classList.toggle('selected',next==='upload');$('tabCounts').setAttribute('aria-selected',next==='counts');$('tabUpload').setAttribute('aria-selected',next==='upload');$('countsSection').hidden=next!=='counts';$('uploadSection').hidden=next!=='upload';clearMessage();}
  function syncTotal(){let sum=0;for(const el of countInputs){const v=Number(el.value);if(Number.isSafeInteger(v)&&v>=0)sum+=v;} $('countsTotal').textContent=sum.toLocaleString('en-US')+(sum===1?' student':' students');}
  function getCounts(){
    if(mode==='upload'){
      if(uploadedCounts===null)throw new Error('Choose a valid CSV file first.');
      return uploadedCounts.slice();
    }
    const counts=countInputs.map(el=>{if(el.value.trim()==='')return NaN;return Number(el.value);});
    FDS.validateCounts(counts);return counts;
  }
  async function receiveFile(file){
    uploadedCounts=null;uploadedFileName='';clearMessage();
    if(!file)return;
    if(file.size>2000000){notify('File is larger than the 2 MB upload limit.');return;}
    try{
      const counts=FDS.parseCsv(await file.text());
      uploadedCounts=counts;uploadedFileName=file.name;
      const n=counts.reduce((a,b)=>a+b,0);
      $('fileFeedback').textContent=`✓ ${file.name} — ${n} valid responses (${counts.join(' / ')} across the five categories)`;
      notify(`Loaded ${n} valid observations. Click “Run statistical analysis”.`,true);
    }catch(e){$('fileFeedback').textContent='Could not read the CSV.';notify(e.message);}
  }
  function htmlDecision(reject,description){return `<div class="decision ${reject?'reject':'retain'}"><strong>${reject?'Reject H₀':'Do not reject H₀'}.</strong> ${description}</div>`;}
  function row(label,value){return `<div class="details-row"><span>${label}</span><b>${value}</b></div>`;}
  function histogram(counts,refCounts){
    const nr=refCounts.reduce((a,b)=>a+b),nn=counts.reduce((a,b)=>a+b);
    return `<div class="chart-shell"><h4>Share of responses by listening-time category</h4>`+
    FDS.LABELS.map((label,i)=>{const r=refCounts[i]/nr*100,v=counts[i]/nn*100;return `<div class="bar-row"><span class="bar-label">${label}</span><div class="bars"><div class="bar-track" title="Reference: ${fmt(r,1)}%"><div class="bar-fill ref" style="width:${r.toFixed(2)}%"></div></div><div class="bar-track" title="New sample: ${fmt(v,1)}%"><div class="bar-fill new" style="width:${v.toFixed(2)}%"></div></div></div></div>`;}).join('')+
    `<div class="legend"><span><i class="ref"></i>Reference (n=${nr})</span><span><i class="new"></i>New sample (n=${nn})</span></div></div>`;
  }
  function tPlot(test){
    const df=test.df,critical=test.critical,observed=test.t;
    const width=560,height=214,left=35,right=15,top=12,bottom=37, xMin=-4.5,xMax=4.5;
    const tx=x=>left+(x-xMin)/(xMax-xMin)*(width-left-right);
    const pdf=x=>Math.exp(FDS.logGamma((df+1)/2)-FDS.logGamma(df/2)-0.5*Math.log(df*Math.PI)-((df+1)/2)*Math.log1p(x*x/df));
    const peak=pdf(0),ty=y=>height-bottom-y/peak*(height-top-bottom)*0.92;
    let path='';for(let i=0;i<=240;i++){const x=xMin+(xMax-xMin)*i/240;path+=(i===0?'M':'L')+tx(x).toFixed(2)+','+ty(pdf(x)).toFixed(2)+' ';}
    const marker=Math.max(xMin,Math.min(xMax,observed)),off=observed!==marker;
    return `<div class="chart-shell"><h4>Welch t distribution (critical boundaries at ±${fmt(critical,2)})</h4><svg class="dist-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="T distribution with critical bounds and the observed t statistic"><rect x="${tx(xMin)}" y="${top}" width="${Math.max(0,tx(-critical)-tx(xMin))}" height="${height-bottom-top}" fill="#fbe7e8"/><rect x="${Math.min(tx(critical),tx(xMax))}" y="${top}" width="${Math.max(0,tx(xMax)-tx(critical))}" height="${height-bottom-top}" fill="#fbe7e8"/><line x1="${left}" x2="${width-right}" y1="${height-bottom}" y2="${height-bottom}" stroke="#c4cde0"/><path d="${path}" fill="none" stroke="#5146cd" stroke-width="2.5"/>${[-4,-2,0,2,4].map(x=>`<text x="${tx(x)}" y="${height-16}" text-anchor="middle" font-size="12" fill="#8996a8">${x}</text>`).join('')}<line x1="${tx(-critical)}" y1="${top}" x2="${tx(-critical)}" y2="${height-bottom}" stroke="#d47d83" stroke-dasharray="4 4"/><line x1="${tx(critical)}" y1="${top}" x2="${tx(critical)}" y2="${height-bottom}" stroke="#d47d83" stroke-dasharray="4 4"/><line x1="${tx(marker)}" y1="${top}" x2="${tx(marker)}" y2="${height-bottom}" stroke="#078f8b" stroke-width="2.7"/><text x="${tx(marker)}" y="11" font-size="11" text-anchor="middle" fill="#057d78">t ${fmt(observed,2)}${off?' (off-scale)':''}</text></svg><div class="legend"><span><i style="background:#d47d83"></i>Rejection regions</span><span><i style="background:#078f8b"></i>Observed t</span></div></div>`;
  }
  function renderReport({counts,welch:w,chi:c}){
    const n=w.newest.n;const warnings=[];
    if(n<10)warnings.push('The new sample has fewer than 10 observations. Approximate inference may be unreliable.');
    if(w.newest.sd<1e-12)warnings.push('Every observation is in one category (zero new-sample variance). Welch’s approximation can be unreliable, especially for very small samples.');
    if(n<=30 && counts.filter(v=>v>0).length<=2)warnings.push('The new sample is concentrated in very few categories. Treat t-test inference cautiously.');
    const meanConclusion=w.reject?'The approximate daily listening-time means differ at the 5% significance level.':'The approximate daily listening-time means do not show a statistically significant difference at the 5% level.';
    const distributionConclusion=c.p<0.05?'The categorical listening-time distributions differ at the 5% significance level.':'There is insufficient evidence of a difference in the categorical listening-time distributions.';
    $('results').innerHTML=`
      <div class="results-title"><div><div class="eyebrow">YOUR RESULTS / LIVE ANALYSIS</div><h2>Statistical report</h2><p>Independent new sample compared with the cleaned FDS reference dataset.</p></div><button type="button" class="outline-btn" id="exportBtn">↓ Download results (.csv)</button></div>
      ${warnings.map(s=>`<div class="report-notice">⚠ ${s}</div>`).join('')}
      <div class="report-summary"><div class="summary-tile"><span>NEW SAMPLE SIZE</span><strong>${n.toLocaleString('en-US')} <small>students</small></strong></div><div class="summary-tile"><span>NEW SAMPLE MEAN (APPROX.)</span><strong>${fmt(w.newest.mean)} <small>hours</small></strong></div><div class="summary-tile"><span>MEAN DIFFERENCE (NEW − REFERENCE)</span><strong>${w.difference>=0?'+':''}${fmt(w.difference)} <small>hours</small></strong></div></div>
      <div class="report-grid"><article class="report-card"><h3>A. Mean listening time</h3><p class="explanation">Two-sided Welch t-test · 95% confidence interval for (new − reference)</p><div class="hypothesis"><b>H₀:</b> μ<sub>new</sub> = μ<sub>reference</sub><br/><b>H₁:</b> μ<sub>new</sub> ≠ μ<sub>reference</sub></div><div class="metric-pair"><div><span>t-STATISTIC</span><strong>${fmt(w.t,4)}</strong></div><div><span>p-VALUE</span><strong>${pFmt(w.p)}</strong></div></div>${htmlDecision(w.reject,meanConclusion)}${tPlot(w)}<div class="details-list">${row('Degrees of freedom',fmt(w.df,2))}${row('95% confidence interval',`[${fmt(w.ciLow)}, ${fmt(w.ciHigh)}] h`)}${row('Critical bounds',`±${fmt(w.critical,3)}`)}${row('New sample SD',`${fmt(w.newest.sd)} h`)}${row('Reference SD',`${fmt(w.reference.sd)} h`)}</div></article>
      <article class="report-card"><h3>B. Listening-category distribution</h3><p class="explanation">Chi-square statistic for a 2 × 5 contingency table</p><div class="hypothesis"><b>H₀:</b> Both samples have the same listening-time category distribution.<br/><b>H₁:</b> The distributions are different.</div><div class="metric-pair"><div><span>χ² STATISTIC</span><strong>${fmt(c.statistic,4)}</strong></div><div><span>p-VALUE</span><strong>${pFmt(c.p)}</strong></div></div>${htmlDecision(c.p<0.05,distributionConclusion)}${histogram(counts,w.reference.counts)}<div class="details-list">${row('Degrees of freedom',c.df)}${row('Minimum expected cell count',fmt(c.minExpected,2))}${row('Method',c.monteCarlo?'Fixed-margin Monte Carlo':'Pearson χ² asymptotic')}${row('Significance level','α = 0.05')}</div>${c.monteCarlo?'<div class="warning">Some expected counts are below 5. A fixed-margin Monte Carlo p-value (10,000 simulated tables, fixed seed) is used instead of the usual chi-square approximation.</div>':''}</article></div><p class="report-smallprint">Comparisons use midpoint approximations for categorical time ranges. Results are exploratory and cannot establish cause-and-effect or describe a wider population without representative sampling. Tests assume independent samples; the original 211-response workbook is identified as synthetic.</p>`;
    $('exportBtn').addEventListener('click',()=>downloadReport({counts,welch:w,chi:c}));
  }
  function downloadReport(r){
    const w=r.welch,c=r.chi;
    const pairs=[['Field','Value'],['Reference count',w.reference.n],['New sample count',w.newest.n],['Reference mean hours',w.reference.mean],['New sample mean hours',w.newest.mean],['Mean difference hours',w.difference],['Welch t',w.t],['Welch df',w.df],['Welch p',w.p],['Welch critical t',w.critical],['CI lower hours',w.ciLow],['CI upper hours',w.ciHigh],['Mean test decision',w.reject?'Reject H0':'Do not reject H0'],['Chi-square',c.statistic],['Chi-square df',c.df],['Distribution p',c.p],['Distribution method',c.method],['Distribution decision',c.p<0.05?'Reject H0':'Do not reject H0'],...FDS.CATEGORIES.map((v,i)=>[v,r.counts[i]])];
    const content='\uFEFF'+pairs.map(line=>line.map(val=>'"'+String(val).replace(/"/g,'""')+'"').join(',')).join('\r\n');
    const link=document.createElement('a');const url=URL.createObjectURL(new Blob([content],{type:'text/csv;charset=utf-8;'}));link.href=url;link.download='FDS_Statistical_Results.csv';document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  $('tabCounts').addEventListener('click',()=>switchMode('counts'));
  $('tabUpload').addEventListener('click',()=>switchMode('upload'));
  countInputs.forEach(el=>el.addEventListener('input',syncTotal));
  $('dropZone').addEventListener('click',()=>$('csvFile').click());
  $('dropZone').addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();$('csvFile').click();}});
  $('csvFile').addEventListener('change',ev=>receiveFile(ev.target.files[0]));
  $('dropZone').addEventListener('dragover',ev=>{ev.preventDefault();$('dropZone').classList.add('dragging');});
  $('dropZone').addEventListener('dragleave',()=>$('dropZone').classList.remove('dragging'));
  $('dropZone').addEventListener('drop',ev=>{ev.preventDefault();$('dropZone').classList.remove('dragging');receiveFile(ev.dataTransfer.files[0]);});
  $('demoBtn').addEventListener('click',()=>{switchMode('counts');[2,5,7,4,2].forEach((v,i)=>countInputs[i].value=v);syncTotal();notify('Demo sample loaded: 20 fictional students. Click “Run statistical analysis”.',true);});
  $('clearBtn').addEventListener('click',()=>{countInputs.forEach(el=>el.value='0');uploadedCounts=null;uploadedFileName='';$('csvFile').value='';$('fileFeedback').textContent='No file selected.';syncTotal();clearMessage();$('results').innerHTML='<div class="empty-state"><div class="empty-icon">▥</div><h2>Your statistical report appears here</h2><p>Enter at least two new observations or upload a valid CSV, then run the analysis.</p></div>';});
  $('runBtn').addEventListener('click',()=>{try{const r=FDS.test(getCounts());lastReport=r;clearMessage();renderReport(r);$('results').scrollIntoView({behavior:'smooth',block:'start'});}catch(e){notify(e.message);}});
  syncTotal();
}
// Module interop for automated numerical validation (doesn't affect browsers).
if(typeof module!=='undefined' && module.exports)module.exports=FDS;
