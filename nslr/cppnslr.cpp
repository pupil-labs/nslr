// Copyleft 2017 Jami Pekkanen <jami.pekkanen@gmail.com>.
// Released under GNU AGPL-3.0, see LICENSE.
/*
<%
cfg['include_dirs'] = [
	'/usr/include/eigen3'
	]
cfg['dependencies'] = ['../segmented_regression.hpp']
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++14', '-g', '-Ofast', '-UNDEBUG', '-Wno-misleading-indentation']
#cfg['compiler_args'] = ['-std=c++14', '-g', '-O0', '-UDEBUG', '-Wno-misleading-indentation']
%>
*/

#include "../segmented_regression.hpp"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;
typedef Segment<Nslr2d::Vector> Segment2d;
typedef Segmentation<Segment2d> Segmentation2d;
PYBIND11_PLUGIN(cppnslr) {
	using namespace pybind11::literals;
	pybind11::module m("cppnslr", "Python NSLR in C++");
	pybind11::class_<Nslr2d>(m, "Nslr2d")
		.def(py::init<Nslr2d::Vector, SplitLikelihood>())
		.def("measurement", py::overload_cast<double, Ref<Nslr2d::Vector>>(&Nslr2d::measurement))
		.def("winner_likelihood", &Nslr2d::winner_likelihood);
	pybind11::class_<Segment2d>(m, "Segment2d")
		.def_readwrite("i", &Segment2d::i)
		.def_readwrite("t", &Segment2d::t)
		.def_readwrite("x", &Segment2d::x);
	pybind11::class_<Segmentation2d>(m, "Segmentation2d")
		.def_readwrite("t", &Segmentation2d::t)
		.def_readwrite("x", &Segmentation2d::x)
		.def_readwrite("segments", &Segmentation2d::segments)
		.def("__call__", py::overload_cast<double>(&Segmentation2d::operator()))
		.def("__call__", py::overload_cast<Timestamps>(&Segmentation2d::operator()));
	m.def("nslr2d", py::overload_cast<Timestamps, Points2d, Ref<const Nslr2d::Vector>>(&nslr2d),
			"ts"_a, "xs"_a, "noise"_a);
	m.def("nslr2d", py::overload_cast<Timestamps, Points2d, double>(&nslr2d),
			"ts"_a, "xs"_a, "noise"_a);
	m.def("nslr2d", py::overload_cast<Timestamps, Points2d, Nslr2d&>(&nslr2d),
		"ts"_a, "xs"_a, "model"_a);
	m.def("nslr2d", py::overload_cast<Timestamps, Points2d, Ref<const Nslr2d::Vector>, double>(&nslr2d),
		"ts"_a, "xs"_a, "noise"_a, "penalty"_a);
	m.def("nslr2d", py::overload_cast<Timestamps, Points2d, double, double>(&nslr2d),
		"ts"_a, "xs"_a, "noise"_a, "penalty"_a);
	m.def("fit_gaze", py::overload_cast<Timestamps, Points2d, double, bool>(&fit_gaze),
		"ts"_a, "xs"_a, "structural_error"_a=0.1, "optimize_noise"_a=true);
	m.def("fit_gaze", py::overload_cast<Timestamps, Points2d, Nslr2d::Vector, bool>(&fit_gaze),
		"ts"_a, "xs"_a, "structural_error"_a, "optimize_noise"_a=true);
	m.def("exponential_split", &exponential_split);
	m.def("penalized_exponential_split", &penalized_exponential_split);
	m.def("constant_penalty_split", &constant_penalty_split);
	m.def("gaze_split", &gaze_split,
			"noise_std"_a,
			"saccade_amplitude"_a=3.0,
			"slow_phase_duration"_a=0.3,
			"slow_phase_speed"_a=5.0
			);
	m.def("fit_2d_segments_cont", &fit_2d_segments_cont);

	return m.ptr();
}
