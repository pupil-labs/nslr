/*
<%
cfg['include_dirs'] = [
	'/usr/include/eigen3'
	]
cfg['dependencies'] = ['segmented_regression.hpp']
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++14', '-fvisibility=hidden', '-Ofast', '-UNDEBUG', '-Wno-misleading-indentation']
#cfg['compiler_args'] = ['-std=c++14', '-g', '-O0', '-UDEBUG', '-Wno-misleading-indentation']
%>
*/

#include "segmented_regression.hpp"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;
typedef Segment<Nslr2d::Vector> Segment2d;
typedef Segmentation<Segment2d> Segmentation2d;
typedef HackSegmentation<Segment2d> HackSegmentation2d;
PYBIND11_PLUGIN(pynslr) {
	using namespace pybind11::literals;
	pybind11::module m("pynslr", "Python NSLR");
	pybind11::class_<Nslr2d>(m, "Nslr2d")
		.def(py::init<Nslr2d::Vector, SplitLikelihood>())
		.def("measurement", py::overload_cast<double, Ref<Nslr2d::Vector>>(&Nslr2d::measurement));
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
	pybind11::class_<HackSegmentation2d>(m, "HackSegmentation2d")
		.def_readwrite("t", &HackSegmentation2d::t)
		.def_readwrite("x", &HackSegmentation2d::x)
		.def_readwrite("segments", &HackSegmentation2d::segments)
		.def("__call__", py::overload_cast<double>(&HackSegmentation2d::operator()))
		.def("__call__", py::overload_cast<Timestamps>(&HackSegmentation2d::operator()));
	m.def("nslr2d", py::overload_cast<Timestamps, Points2d, Nslr2d&>(&nslr2d),
		"ts"_a, "xs"_a, "model"_a);
	m.def("nslr2d", py::overload_cast<Timestamps, Points2d, Ref<const Nslr2d::Vector>, double>(&nslr2d),
		"ts"_a, "xs"_a, "noise"_a, "penalty"_a=5.0);
	m.def("nslr2d", py::overload_cast<Timestamps, Points2d, double, double>(&nslr2d),
		"ts"_a, "xs"_a, "noise"_a, "penalty"_a=5.0);
	m.def("exponential_split", &exponential_split);
	m.def("penalized_exponential_split", &penalized_exponential_split);
	m.def("fit_2d_segments_cont", &fit_2d_segments_cont);
	return m.ptr();
}
