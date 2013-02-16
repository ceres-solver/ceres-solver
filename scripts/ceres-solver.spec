Name:           ceres-solver
Version:        1.4.0
Release:        0%{?dist}
Summary:        A non-linear least squares minimizer

Group:          Development/Libraries
License:        BSD
URL:            http://code.google.com/p/ceres-solver/
Source0:        http://%{name}.googlecode.com/files/%{name}-%{version}.tar.gz
BuildRoot:      %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

%if (0%{?rhel} <= 6)
BuildRequires:  cmake28
%else
BuildRequires:  cmake
%endif
BuildRequires:  eigen3-devel
BuildRequires:  suitesparse-devel
BuildRequires:  blas-devel 
BuildRequires:  lapack-devel
BuildRequires:  protobuf-devel
BuildRequires:  gflags-devel
BuildRequires:  glog-devel

%description
Ceres Solver is a portable C++ library that allows for modeling and solving
large complicated nonlinear least squares problems. Features:

  - A friendly API: build your objective function one term at a time
  - Automatic differentiation
  - Robust loss functions
  - Local parameterizations
  - Threaded Jacobian evaluators and linear solvers
  - Levenberg-Marquardt and Dogleg (Powell & Subspace) solvers
  - Dense QR and Cholesky factorization (using Eigen) for small problems
  - Sparse Cholesky factorization (using SuiteSparse) for large sparse problems
  - Specialized solvers for bundle adjustment problems in computer vision
  - Iterative linear solvers for general sparse and bundle adjustment problems
  - Runs on Linux, Windows, Mac OS X and Android. An iOS port is underway


%package        devel
Summary:        A non-linear least squares minimizer
Group:          Development/Libraries
Requires:       %{name} = %{version}-%{release}

%description    devel
The %{name}-devel package contains libraries and header files for
developing applications that use %{name}.


%prep
%setup -q

%build
mkdir build
pushd build
%if (0%{?rhel} <= 6)
%{cmake28} ..
%else
%{cmake} ..
%endif
make %{?_smp_mflags}


%install
rm -rf $RPM_BUILD_ROOT
pushd build
make install DESTDIR=$RPM_BUILD_ROOT
find $RPM_BUILD_ROOT -name '*.la' -delete


%clean
rm -rf $RPM_BUILD_ROOT


%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig


%files
%defattr(-,root,root,-)
%doc
%{_libdir}/*.so.*

%files devel
%defattr(-,root,root,-)
%doc
%{_includedir}/*
%{_libdir}/*.so
%{_libdir}/*.a


%changelog
* Sun Oct 14 2012 Taylor Braun-Jones <taylor@braun-jones.org> - 1.4.0-0
- Initial creation
