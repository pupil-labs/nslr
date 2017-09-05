#include <vector>
#include <unordered_set>
#include <list>
#include <tuple>
#include <forward_list>
#include <deque>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cmath>

#if !EIGEN_VERSION_AT_LEAST(3,3,0)
#error "Needs eigen version of at least 3.3.0"
#endif

typedef unsigned int uint;
using Eigen::Array;
using Eigen::Map;
using Eigen::Ref;

template <typename T>
struct SharedList {
	struct SharedNode {
		SharedNode *parent = NULL;
		uint refcount;
		T value;

		SharedNode(SharedNode *parent, T value)
			:parent(parent), refcount(1), value(value)
		{
			if(parent) {
				parent->refcount += 1;
			}
		}

		SharedNode(const SharedNode& other) {
			std::cout << "Copying!" << std::endl;
		}

		~SharedNode() {
			if(parent) {
				parent->refcount -= 1;
			}
		}
	};

	SharedNode *tail;
	
	SharedList() {
		tail = NULL;
	}

	SharedList(const SharedList &parent, T value)
	{
		tail = new SharedNode(parent.tail, value);
	}
	
	SharedList(const SharedList& that) {
		tail = that.tail;
		if(tail) {
			tail->refcount += 1;
		}
	}
	
	SharedList(SharedList&& that) {
		tail = that.tail;
		that.tail = NULL;
	}

	SharedList& operator=(SharedList&& that) {
		if(this == &that) return *this;
		std::swap(tail, that.tail);
		return *this;
	}

	~SharedList() {
		if(!tail) return;
		tail->refcount -= 1;
		
		// We could in theory do this cascade in the
		// destructor of the SharedNode. However, in practice
		// even on realistic data that leads to a huge recursion that
		// leads to a stack overflow. And this is probably a lot faster
		// anyway.
		auto node = tail;
		while(node and node->refcount == 0) {
			auto killme = node;
			node = node->parent;
			delete killme;
		}
	}
};


/* Naive optimal linear segmentation.
   Assumes that velocity between segments can be
   "infinite" (ie start of the segment doesn't in any way depend from the
   previous). Not very realistic, but statistically a simple case.
*/
/*
template <class Vector, class Model>
struct NslrHypothesis {
	uint n = 0;
	double t = 0.0;
	double ws = 0.0;
	double mean_t = 0.0;
	double ss_t = 0.0;
	Vector mean_x;
	Vector ss_x;
	Vector ss_xt;
	Vector residual_ss;
	double _total_likelihood = 0.0;
	
	// This should probably be const Model* const, but STL tears a new one
	// if we declare it so, as it will kill the move constructor.
	// Actually it should be a reference, but STL really really doesn't
	// like that. This is one horrible language.
	const Model* model;
	SharedList<uint> splits;
	//double segment_lik = 0.0;
	
	void __initialize_variables() {
		// C++ constructors suck.
		t *= 0;
		mean_t *= 0;
		mean_x.setZero();
		ss_x.setZero();
		ss_xt.setZero();
		residual_ss.setZero();
	}
	
	NslrHypothesis(const Model* model, NslrHypothesis& parent, double dt, uint i)
		:model(model), splits(parent.splits, i)
	{
		_total_likelihood = parent.likelihood();
		_total_likelihood += model->split_likelihood(dt);
		__initialize_variables();
	}
	
	NslrHypothesis(const Model* model)
		:model(model)
	{
		//splits = decltype(splits)(splits, 0);
		__initialize_variables();
	}
	
	
	inline void measurement(double dt, double* position, double w=1.0) {
		measurement(dt, Map<Vector>(position), w);
	}
	
	inline void measurement(double dt, Ref<Vector> position, double w=1.0) {
		n++;
		ws += w;
		double wsinv = w/ws;
		double ninv = 1.0/n;
		auto delta_x = (position - mean_x).eval();
		mean_x += (delta_x*wsinv);
		ss_x += w*delta_x.cwiseProduct(position-mean_x);
		
		t += dt;
		auto delta_t = t - mean_t;
		mean_t += delta_t*wsinv;
		ss_t += w*delta_t*(t - mean_t);

		// Calculate the regression SS incrementally
		// (for both independent axes simultaneously)
		//ss_xt += ((n-1)*ninv)*delta_x*delta_t;
		ss_xt += ((ws - w)*wsinv)*delta_x*delta_t;
		
		auto new_residual_ss = (ss_x - ss_xt.pow(2)/ss_t).eval();
		if(ss_t == 0) new_residual_ss = ss_x; // Sometimes zero by zero is zero
		
		_total_likelihood += ((residual_ss - new_residual_ss)*model->resid_normer).sum() + model->seg_normer;
		residual_ss = new_residual_ss;
		
	}

	inline double likelihood() const {
		return _total_likelihood;
	}

	Vector predict(double nt) {
		if(ss_t == 0) {
			return mean_x;
		}
		auto slope = ss_xt/ss_t;
		auto intercept = mean_x - slope*mean_t;
		return nt*slope + intercept;
	}
};*/


template <class Vector, class Model>
struct NslrHypothesis {
	//(Stt*Sww*Sxx - Stt*Swx**2 - Stw**2*Sxx + 2*Stw*Stx*Swx - Stx**2*Sww)/(Stt*Sww - Stw**2)
	double t = 0.0;

	double Stt = 0.0;
	double Sww = 0.0;
	Vector Sxx;
	Vector Swx;
	double Stw = 0.0;
	Vector Stx;
	
	uint n = 0;
	
	Vector residual_ss;
	double _total_likelihood = 0.0;

	bool has_parent;
	Vector startpoint;

	// This should probably be const Model* const, but STL tears a new one
	// if we declare it so, as it will kill the move constructor.
	// Actually it should be a reference, but STL really really doesn't
	// like that. This is one horrible language.
	const Model* model;
	SharedList<uint> splits;
	//double segment_lik = 0.0;
	
	void __initialize_variables() {
		// C++ constructors suck.
		Sxx.setZero();
		Swx.setZero();
		Stx.setZero();
		residual_ss.setZero();
	}
	
	NslrHypothesis(const Model* model, NslrHypothesis& parent, double dt, uint i)
		:model(model), splits(parent.splits, i)
	{
		__initialize_variables();
		startpoint = parent.predict(parent.t);
		has_parent = true;
		_total_likelihood = parent.likelihood();
		_total_likelihood += model->split_likelihood(dt);
	}
	
	NslrHypothesis(const Model* model)
		:model(model)
	{
		//splits = decltype(splits)(splits, 0);
		__initialize_variables();
		has_parent = false;
	}
	
	
	inline void measurement(double dt, double* position, double w=1.0) {
		measurement(dt, Map<Vector>(position), w);
	}
	
	inline void measurement(double dt, Ref<Vector> position, double w=1.0) {
		//(Stt*Sww*Sxx - Stt*Swx**2 - Stw**2*Sxx + 2*Stw*Stx*Swx - Stx**2*Sww)/(Stt*Sww - Stw**2)
		// Stt, Sww, Sxx, Swx, Stw, Stx
		n += 1;
		
		t += dt;
		Vector x = w*position;
		double t_ = w*t;

		Stt += t_*t_;
		Sww += w*w;
		Sxx += x*x;
		Swx += w*x;
		Stw += t_*w;
		Stx += t_*x;
		
		Vector b;
		if(has_parent) {
			b = startpoint;
		} else {
			b = (Stt*Swx - Stw*Stx)/(Stt*Sww - Stw*Stw);
		}
		Vector a = (Stx - b*Stw)/Stt;
		//Vector a = (-Stw*Swx + Stx*Sww)/denom;
		Vector new_residual_ss = a*a*Stt + 2*a*b*Stw - 2*a*Stx + b*b*Sww - 2*b*Swx + Sxx;
		//Vector new_residual_ss =
		//	(Stt*Sww*Sxx - Stt*Swx.pow(2) - Stw*Stw*Sxx + 2*Stw*Stx*Swx - Stx.pow(2)*Sww)/(Stt*Sww - Stw*Stw);
		//	//(Mtt*Mww*Mxx - Mtt*Mwx.pow(2) - Mtw*Mtw*Mxx + 2*Mtw*Mtx*Mwx - Mtx.pow(2)*Mww)/(Mtt*Mww - Mtw**2)
		
		//auto new_residual_ss = (ss_x - ss_xt.pow(2)/ss_t).eval();
		if(n < 2) new_residual_ss.setZero(); // Sometimes zero by zero is zero
		
		_total_likelihood += ((residual_ss - new_residual_ss)*model->resid_normer).sum() + model->seg_normer;
		residual_ss = new_residual_ss;
		
	}

	inline double likelihood() const {
		return _total_likelihood;
	}

	Vector predict(double nt) {
		Vector b;
		if(has_parent) {
			b = startpoint;
		} else {
			b = (Stt*Swx - Stw*Stx)/(Stt*Sww - Stw*Stw);
		}
		Vector a = (Stx - b*Stw)/Stt;
		return nt*a + b;
	}
};


typedef std::function<double(double)> SplitLikelihood;

SplitLikelihood exponential_split(double split_rate) {
	return [=](double dt) {
		return log(1.0-exp(-split_rate*dt));
	};
}

SplitLikelihood penalized_exponential_split(double penalty) {
	return [=](double dt) {
		return log(dt) - penalty;
	};
}

SplitLikelihood constant_penalty_split(double penalty) {
	return [=](double dt) {
		return -penalty;
	};
}

/*
======================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------
Intercept             -2.8614      0.520     -5.504      0.000      -3.882      -1.841
log(sampling_rate)    -1.0397      0.095    -10.953      0.000      -1.226      -0.853
I(1 / noise_level)     0.4993      0.013     39.175      0.000       0.474       0.524
log(max_distance)      0.5149      0.064      8.030      0.000       0.389       0.641
log(max_speed)         0.1118      0.062      1.801      0.072      -0.010       0.234
log(max_duration)     -0.9375      0.078    -11.962      0.000      -1.091      -0.784
======================================================================================
*/ 
SplitLikelihood gaze_split(
		double noise_std,
		double saccade_amplitude=3.0,
		double slow_phase_duration=0.3,
		double slow_phase_speed=5.0) {
	using std::log; using std::exp;
	/*double pinc =
		1.2*std::log(noise_std) +
		-0.6*std::log(saccade_amplitude/2.0) +
		1.0*std::log(slow_phase_duration/2.0) +
		-0.15*std::log(slow_phase_speed/2.0) +
		2.3;
	*/
	double logit_pinc = 
		0.5*(1.0/noise_std) +
		0.5*log(saccade_amplitude*2) +
		-1.0*log(slow_phase_duration*2) +
		0.1*log(slow_phase_speed*2) +
		-3.0;
	return [=](double dt) {
		double lp = logit_pinc + -1.0*log(1/dt);
		return log(1/(1 + exp(-lp)));
		//return -(1.0*std::log(1/dt) + pinc);
	};

}

template <uint ndim>
struct Nslr {
	using Vector = Array<double, ndim, 1>;
	using Hypothesis = NslrHypothesis<Vector, Nslr>;
	Vector noise_std;
	Vector noise_prec;
	SplitLikelihood _split_likelihood;
	//double split_rate;
	std::vector<Hypothesis, Eigen::aligned_allocator<Hypothesis>> hypotheses;
	
	double seg_normer; // An optimization to avoid taking logs in the loop
	double split_compensation;
	Vector resid_normer;

	uint i = 0;

       	Nslr(Vector noise_std, double split_rate) : Nslr(noise_std, exponential_split(split_rate)) {}


        Nslr(Vector noise_std, SplitLikelihood splitter)
		: noise_std(noise_std), _split_likelihood(splitter)
	{
		seg_normer = (1.0/(noise_std*std::sqrt(2*M_PI))).log().sum();
		resid_normer = 1.0/(2*noise_std.pow(2));
		
		//split_compensation = (noise_std.pow(2)*resid_normer).sum() + seg_normer;
		split_compensation = 0.0;
        }

	auto split_likelihood(double dt) const {
		return _split_likelihood(dt) - split_compensation;
	}
	
	void measurement(double dt, double *position) {
		Map<Vector> pos(position);
		measurement(dt, pos);
	}

	Hypothesis& get_winner() {
		static const auto likcmp = [](const Hypothesis& a, const Hypothesis& b) {
			//if(a.n < 2 && b.n < 2) return true; 
			//if(a.n >= 2 && b.n < 2) return false; 
			//if(a.n < 2 && b.n >= 2) return true; 
			return a.likelihood() < b.likelihood();
		};
		
		return *std::max_element(hypotheses.begin(), hypotheses.end(), likcmp);
	}

	auto winner_likelihood() {
		return get_winner().likelihood();
	}

	void measurement(double dt, Ref<Vector> measurement) {
		if(hypotheses.empty()) {
			hypotheses.emplace_back(this);
			auto& root = hypotheses.back();
			root.measurement(dt, measurement);
			++i;
			return;
		}
		
		for(auto it=hypotheses.begin(); it < hypotheses.end(); ++it) {
			it->measurement(dt, measurement);
		}

		auto& winner = get_winner();
		hypotheses.emplace_back(this, winner, dt, i);
		auto& new_hypo = hypotheses.back();
		//new_hypo.measurement(dt, measurement);
		Vector pred = winner.predict(winner.t);
		//new_hypo.measurement(0.0, pred, 1e6);
		auto worst_survivor = new_hypo.likelihood();
		
		const auto no_chance = [&](const Hypothesis& hypo) {
			if(&hypo == &winner) {
				return false;
			}
			return hypo.likelihood() <= worst_survivor;
		};
		
		auto erased_start = std::remove_if(hypotheses.begin(), hypotheses.end() - 1, no_chance);
		hypotheses.erase(erased_start, hypotheses.end() - 1);
		
		
		++i;
	}
};



/*struct Nslr {
	
}*/

typedef Nslr<2u> Nslr2d;
typedef Ref<const Array<double, -1, 1>> Timestamps;
typedef Ref<const Array<double, -1, 2>> Points2d;
typedef const std::vector<size_t>& Splits;

template <class Vector_>
struct Segment {
	using Vector = Vector_; // This is a horrible language!
	std::tuple<size_t, size_t> i;
	std::tuple<double, double> t;
	std::tuple<Vector, Vector> x;
};

template <class Segment>
struct HackSegmentation {
	using Vector = typename Segment::Vector;
	using Vectors = Array<double, -1, Vector::RowsAtCompileTime>;
	//static auto g = std::get;
	std::vector<Segment> segments;
	std::vector<double> t;
	std::vector<Vector> x;
	HackSegmentation(Timestamps ts, std::vector<Segment> segments) :segments(segments) {
		std::list<std::tuple<double, Vector, size_t>> points;
		auto& s = segments[0];
		points.emplace_back(std::get<0>(s.t), std::get<0>(s.x), std::get<1>(s.i) - std::get<0>(s.i));
		for(auto& s : segments) {
			auto t0 = std::get<0>(s.t);
			auto t1 = std::get<1>(s.t);
			auto x0 = std::get<0>(s.x);
			auto x1 = std::get<1>(s.x);
			auto i0 = std::get<0>(s.i);
			auto i1 = std::get<1>(s.i);

			auto dur = t1 - t0;
			auto n = i1 - i0;
			auto f = [&](auto t) {
				auto w = (t - t0)/dur;
				return w*x1 + (1.0 - w)*x0;
			};
			
			// TODO: The points get put wrong way around!?
			auto& pp = points.back();
			auto px = std::get<1>(pp);
			auto pn = std::get<2>(pp);
			// Should be MLE-estimate using the segment regressions,
			// but a weighted average'll do for now.
			std::get<1>(pp) = (px*(pn - 2 + 1e-6) + std::get<0>(s.x)*(n - 2 + 1e-6))/(pn + n - 4 + 2e-6);
			std::get<2>(pp) += n;

			if(n > 2) {
				auto nt = ts(i0 + 1, 0);
				points.emplace_back(nt, f(nt), n);
			}
			if(n > 3) {
				auto nt = ts(i1 - 2, 0);
				points.emplace_back(nt, f(nt), n);

			}

			points.emplace_back(t1, x1, n);
		}
		
		for(auto& p : points) {
			t.push_back(std::get<0>(p));
			x.push_back(std::get<1>(p));
		}
	}

	Vector operator()(double nt) {
		auto idx = std::distance(t.begin(), std::lower_bound(t.begin(), t.end(), nt)) - 1;
		idx = std::min(idx, decltype(idx)(t.size() - 2));
		idx = std::max(idx, decltype(idx)(0));
		auto t0 = t[idx];
		auto t1 = t[idx+1];
		auto x0 = x[idx];
		auto x1 = x[idx+1];
		auto w = (nt - t0)/(t1 - t0);
		return x1*w + x0*(1.0 - w);
	}
	
	Vectors operator()(Timestamps nts) {
		Vectors out(nts.rows(), decltype(nts.rows())(Vectors::ColsAtCompileTime));
		for(size_t i=0; i < nts.rows(); ++i) {
			out.row(i) = (*this)(nts(i, 0)).transpose();
		}
		return out;
	}
};

template <class Segment>
struct Segmentation {
	using Vector = typename Segment::Vector;
	using Vectors = Array<double, -1, Vector::RowsAtCompileTime>;
	//static auto g = std::get;
	std::vector<Segment> segments;
	std::vector<double> t;
	std::vector<Vector> x;
	Segmentation(Timestamps ts, std::vector<Segment> segments) :segments(segments) {
		for(auto &s : segments) {
			t.push_back(std::get<0>(s.t));
			x.push_back(std::get<0>(s.x));
		}
		auto &s = segments.back();
		t.push_back(std::get<1>(s.t));
		x.push_back(std::get<1>(s.x));
	}

	Vector operator()(double nt) {
		auto idx = std::distance(t.begin(), std::lower_bound(t.begin(), t.end(), nt)) - 1;
		idx = std::min(idx, decltype(idx)(t.size() - 2));
		idx = std::max(idx, decltype(idx)(0));
		auto t0 = t[idx];
		auto t1 = t[idx+1];
		auto x0 = x[idx];
		auto x1 = x[idx+1];
		auto w = (nt - t0)/(t1 - t0);
		return x1*w + x0*(1.0 - w);
	}
	
	Vectors operator()(Timestamps nts) {
		Vectors out(nts.rows(), decltype(nts.rows())(Vectors::ColsAtCompileTime));
		for(size_t i=0; i < nts.rows(); ++i) {
			out.row(i) = (*this)(nts(i, 0)).transpose();
		}
		return out;
	}
};


auto fit_2d_linear_segments(Timestamps ts, Points2d xs, Splits splits) {
	//Array<double, -1, 1> sts(n_segments*2);
	//Array<double, -1, 2> sxs(n_segments*2, 2);
	std::vector<Segment<Nslr2d::Vector>> segments;
	auto n = ts.rows();

	auto add_segment = [&](auto i, auto start, auto end) {
		// end is inclusive, except in the very end
		if(end < n) {
			end += 1;
		}
		auto len = end - start;
		auto ets = ts(end - 1, 0);
		auto t = ts.block(start, 0, len, 1);
		auto x = xs.block(start, 0, len, 2);
		
		auto tmean = t.colwise().mean();
		auto xmean = x.colwise().mean();
		auto tcent = t.rowwise() - tmean;
		auto xcent = x.rowwise() - xmean;
		auto tss = (tcent*tcent).colwise().mean();
		
		Array<double, -1, 2> cos = xcent;

		// Can't get eigen to broadcast :(
		cos.col(0) *= tcent;
		cos.col(1) *= tcent;
		auto coss = cos.colwise().mean();
		auto slope = coss/tss(0, 0);
		auto intercept = xmean - slope*tmean(0, 0);

		/*sts.row(i*2) = t.row(0);
		sxs.row(i*2) = slope*t(0, 0) + intercept;
		
		sts.row(i*2+1) = ets;
		sxs.row(i*2+1) = slope*ets + intercept;*/
		Segment<Nslr2d::Vector> segment = {
			.i=std::make_tuple(start, end),
			.t=std::make_tuple(t(0,0), ets),
			.x=std::make_tuple(
				slope*t(0, 0) + intercept,
				slope*ets + intercept
			)
		};
		//Segment<Nslr2d::Vector> segment;
		return segment;
		//return segment;
		//sts.push_back(ets);
		//sxs.push_back(slope*ets + intercept);
	};
	
	for(size_t i = 0; i < splits.size() - 1; ++i) {
		segments.push_back(add_segment(i, splits[i], splits[i+1]));
	}
	return Segmentation<Segment<Nslr2d::Vector>>(ts, segments);
	//return std::make_tuple(sts, sxs);

}

template<typename Tt, typename Tx>
struct TridiagonalSolver {
	std::list<std::tuple<Tx, Tt>> BG;
	//B;
	//std::list<T> G;

	TridiagonalSolver() {
		BG.emplace_back(0.0, 0.0);
	}

	void add_row(Tt t0, Tt t1, Tt t2, Tx x) {
		/*std::cout
			<< t1 << '\t'
			<< t1 << '\t'
			<< t2
			<< std::endl;*/
		Tx b; Tt g;
		std::tie(b, g) = BG.back();
		//auto b = B.back();
		//auto g = G.back();
		auto denom = t0*g + t1;
		BG.emplace_back((x - t0*b)/denom, -t2/denom);
	}

	std::list<Tx> solve() {
		Tx x(0.0);
		std::list<Tx> X;
		auto first = BG.cbegin();
		for(auto bg = --BG.cend(); bg != first; --bg) {
			Tx b; Tt g;
			std::tie(b, g) = *bg;
			x = g*x + b;
			X.emplace_front(x);
		}

		return X;
	}
};



auto fit_2d_segments_cont(Timestamps ts, Points2d xs, Splits splits) {
	using Point = Nslr2d::Vector;
	TridiagonalSolver<double, Array<double, 1, 2>> solver;
	auto n_segments = splits.size() - 1;
	double Mmw0=0.0, Mww0=0.0, Mmw1=0.0, Mmm1=0.0, p0=0.0, p1=0.0, p2=0.0;
	//double Mmm0, Mww1 = 0.0;
	Point Mxw0(0.0);
	Point Mxm0(0.0);
	Point y;

	for(size_t i = 0; i < n_segments; ++i) {
		auto start = splits[i];
		auto end = splits[i+1];
		auto len = end - start;
		//std::cout << "Start " << start << " last " << end - 1 << " len " << ts.size() << std::endl;
		double ets = ts(end - 1, 0);
		//std::cout << "Got it" << std::endl;
		double sts = ts(start, 0);
		auto t = ts.block(start, 0, len, 1);
		auto x = xs.block(start, 0, len, 2);
		auto dur = ets - sts;
		if(dur == 0) dur = 1.0;
		auto w = (t - sts)/dur;
		auto m = 1 - w;
		Mmw1 = (m*w).sum();
		Mmm1 = (m*m).sum();
		//std::cout << "Vec " << std::endl << (x.colwise()*m).transpose() << std::endl;
		//std::cout << "Sum " << (x.colwise()*m).colwise().sum() << std::endl;
		Point Mxm1((x.colwise()*m).colwise().sum());
		
		p0 = Mmw0;
		p1 = Mmm1 + Mww0;
		p2 = Mmw1;
		Point y = Mxm1 + Mxw0;
		solver.add_row(p0, p1, p2, y);
		//std::cout << double(p0(0,0)) << " " << double(p1(0,0)) << " " << double(p2(0,0)) << " " << double(y(0,0)) << std::endl;
		//std::cout << "Ys " << y.transpose() << std::endl;
		
		Mmw0 = Mmw1;
		//Mmm0 = Mmm1;
		Mxm0 = Mxm1;
		Mww0 = (w*w).sum();
		Mxw0 = (x.colwise()*w).colwise().sum();
	}

	p0 = Mmw0;
	p1 = Mww0;
	p2 = 0.0;
	y = Mxw0;
	solver.add_row(p0, p1, p2, y);
	//std::cout << double(p0(0,0)) << " " << double(p1(0,0)) << " " << double(p2(0,0)) << " " << double(y(0,0)) << std::endl;
	auto point_list = solver.solve();
	// Feeling lazy
	std::vector<Segment<Point>> segments;
	std::vector<Point> points(point_list.begin(), point_list.end());
	size_t last_idx = ts.rows() - 1;
	for(size_t j = 0; j < n_segments; ++j) {
		auto i0 = splits[j];
		auto i1 = splits[j+1];
		auto t0 = ts(i0, 0);
		auto t1 = ts(std::min(i1, last_idx), 0);
		auto r0 = points[j];
		auto r1 = points[j+1];
		
		Segment<Point> segment = {
			.i=std::make_tuple(i0, i1),
			.t=std::make_tuple(t0, t1),
			.x=std::make_tuple(r0, r1)
		};
		segments.push_back(segment);
	}

	return Segmentation<Segment<Point>>(ts, segments);
}

template <typename T>
auto colstd(T xs) {
	auto mean = xs.colwise().mean();
	auto err = (xs.rowwise() - mean);
	auto std = (err*err).colwise().mean().sqrt();
	return std;
}

auto nslr2d(Timestamps ts, Points2d xs, Nslr2d& model) {
	size_t n = ts.rows();
	auto prev_t = ts(0, 0);
	for(size_t i = 0; i < n; ++i) {
		double dt = ts(i, 0) - prev_t;
		prev_t = ts(i, 0);
		Nslr2d::Vector x(xs.row(i));
		model.measurement(dt, x);
	}
	
	
	/*
	std::vector<size_t> splits;
	auto split = model.get_winner().splits.tail;
	while(split) {
		auto next_split = split->value;
		splits.push_back(next_split);
		split = split->parent;
	}
	splits.push_back(0);
	std::reverse(splits.begin(), splits.end());
	splits.push_back(ts.rows());
	return fit_2d_linear_segments(ts, xs, splits);
	*/
	
	std::vector<size_t> splits;
	auto split = model.get_winner().splits.tail;
	while(split) {
		auto next_split = split->value;
		splits.push_back(next_split);
		split = split->parent;
	}
	splits.push_back(0);
	std::reverse(splits.begin(), splits.end());
	splits.push_back(ts.rows());
	return fit_2d_segments_cont(ts, xs, splits);
}

auto nslr2d(Timestamps ts, Points2d xs, Ref<const Nslr2d::Vector> noise, double penalty) {
	Nslr2d model(noise, penalized_exponential_split(penalty));
	return nslr2d(ts, xs, model);
}

auto nslr2d(Timestamps ts, Points2d xs, Ref<const Nslr2d::Vector> noise) {
	Nslr2d model(noise, gaze_split(noise.mean()));
	return nslr2d(ts, xs, model);
}

auto nslr2d(Timestamps ts, Points2d xs, double noise) {
	Nslr2d::Vector noise_vec(noise);
	return nslr2d(ts, xs, noise_vec);
}

auto nslr2d(Timestamps ts, Points2d xs, double noise, double penalty) {
	Nslr2d::Vector noise_vec;
	noise_vec.setConstant(noise);
	return nslr2d(ts, xs, noise_vec, penalty);
}

// Hash function for Eigen matrix and vector.
// The code is from `hash_combine` function of the Boost library. See
// http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
template<typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    // Note that it is oblivious to the storage order of Eigen matrix (column- or
    // row-major). It will give you the same hash value for two different matrices if they
    // are the transpose of each other in different storage order.
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};



//auto fit_gaze(Timestamps ts, Points2d xs, double minerr=0.5) {
//
//}

auto nslr2d(Timestamps ts, Points2d xs, std::function<Nslr2d(Nslr2d::Vector)> getmodel, Nslr2d::Vector structural_error) {
	Nslr2d::Vector nl = colstd(xs);
	std::unordered_set<decltype(nl.matrix()), matrix_hash<decltype(nl.matrix())>> seen;
	seen.insert(nl.matrix());
	while(true) {
		nl += structural_error;
		auto model = getmodel(nl);
		if(nl.prod() == 0) {
			nl = nl.unaryExpr([](double a) -> double { return std::max(a, 1e-9); });
			model = getmodel(nl);
			return nslr2d(ts, xs, model);
		}
		auto fit = nslr2d(ts, xs, model);
		auto error = fit(ts) - xs;
		nl = colstd(error);
		
		if(seen.find(nl.matrix()) != seen.end()) {
			return fit;
		}
		
		seen.insert(nl.matrix());
	}
}

auto fit_gaze(Timestamps ts, Points2d xs, Nslr2d::Vector structural_error, bool optimize_noise=true) {
	if(!optimize_noise) {
		Nslr2d model(structural_error, gaze_split(structural_error.mean()));
		return nslr2d(ts, xs, model);
	}
	auto getmodel = [&](auto nl) {
		return Nslr2d(nl, gaze_split(nl.mean()));
	};
	return nslr2d(ts, xs, getmodel, structural_error);
}

auto fit_gaze(Timestamps ts, Points2d xs, double structural_error=0.1, bool optimize_noise=true) {
	Nslr2d::Vector se(structural_error);
	return fit_gaze(ts, xs, se, optimize_noise);
}

/*auto fit_gaze(Timestamps ts, Points2d xs, double structural_error=0.5, bool optimize_noise=true) {
	return fit_gaze(ts, xs, structural_error, optimize_noise);
}*/
