function [spikeMat, tVec] = poissonSpikeGen(fr, tSim, nTrials)
dt = 1/10000;
nBins = floor(tSim/dt);
spikeMat = rand(nTrials, nBins) < fr*dt;
tVec = 0:dt:tSim-dt;
end