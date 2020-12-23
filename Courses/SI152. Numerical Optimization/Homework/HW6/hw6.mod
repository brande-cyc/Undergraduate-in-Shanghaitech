# Define set: begin city and destination
set begin;
set dest;
# Define parameters: cost, population, distance<=150
param c {dest};
param p {begin};
param d {begin, dest};
param alpha;
# Define decision variables: DC_location, being_covered_or_not
var x{dest} binary;
var g{begin} binary;

# objective
minimize COST:
	sum {i in dest} c[i] * x[i];
# constraint 1: alpha cover rate
subject to COVER_RATE:
	(sum {i in begin} g[i] * p[i]) >= (alpha * sum {j in begin} p[j]);
# constraint 2 & 3: Switching constraint
subject to SWITCH_1 {i in begin}:
	g[i] <= sum {j in dest} x[j] * d[i, j];
subject to SWITCH_2 {i in begin}:
	g[i] >= sum {j in dest} x[j] * d[i, j] / 1000;
	
