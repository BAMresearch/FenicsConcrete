Help on Function in module dolfinx.fem.function object:

class FFuunnccttiioonn(ufl.coefficient.Coefficient)
 |  Function(V: 'FunctionSpace', x: 'typing.Optional[la.VectorMetaClass]' = None, name: 'typing.Optional[str]' = None, dtype: 'np.dtype' = <class 'numpy.float64'>)
 |  
 |  A finite element function that is represented by a function space
 |  (domain, element and dofmap) and a vector holding the
 |  degrees-of-freedom
 |  
 |  Method resolution order:
 |      Function
 |      ufl.coefficient.Coefficient
 |      ufl.core.terminal.FormArgument
 |      ufl.core.terminal.Terminal
 |      ufl.core.expr.Expr
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  ____iinniitt____(self, V: 'FunctionSpace', x: 'typing.Optional[la.VectorMetaClass]' = None, name: 'typing.Optional[str]' = None, dtype: 'np.dtype' = <class 'numpy.float64'>)
 |      Initialize a finite element Function.
 |      
 |      Args:
 |          V: The function space that the Function is defined on.
 |          x: Function degree-of-freedom vector. Typically required
 |              only when reading a saved Function from file.
 |          name: Function name.
 |          dtype: Scalar type.
 |  
 |  ____ssttrr____(self)
 |      Pretty print representation of it self.
 |  
 |  ccoollllaappssee(self) -> 'Function'
 |  
 |  ccooppyy(self) -> 'Function'
 |      Return a copy of the Function. The FunctionSpace is shared and the
 |      degree-of-freedom vector is copied.
 |  
 |  eevvaall(self, x: 'npt.ArrayLike', cells: 'npt.ArrayLike', u=None) -> 'np.ndarray'
 |      Evaluate Function at points x, where x has shape (num_points, 3),
 |      and cells has shape (num_points,) and cell[i] is the index of the
 |      cell containing point x[i]. If the cell index is negative the
 |      point is ignored.
 |  
 |  iinntteerrppoollaattee(self, u: 'typing.Union[typing.Callable, Expression, Function]', cells: 'typing.Optional[np.ndarray]' = None) -> 'None'
 |      Interpolate an expression
 |      
 |      Args:
 |          u: The function, Expression or Function to interpolate.
 |          cells: The cells to interpolate over. If `None` then all
 |              cells are interpolated over.
 |  
 |  sspplliitt(self) -> 'tuple[Function, ...]'
 |      Extract any sub functions.
 |      
 |      A sub function can be extracted from a discrete function that
 |      is in a mixed, vector, or tensor FunctionSpace. The sub
 |      function resides in the subspace of the mixed space.
 |      
 |      Args:
 |          Function space subspaces.
 |  
 |  ssuubb(self, i: 'int') -> 'Function'
 |      Return a sub function.
 |      
 |      Args:
 |          i: The index of the sub-function to extract.
 |      
 |      Note:
 |          The sub functions are numbered i = 0..N-1, where N is the
 |          total number of sub spaces.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties defined here:
 |  
 |  ddttyyppee
 |  
 |  ffuunnccttiioonn__ssppaaccee
 |      The FunctionSpace that the Function is defined on
 |  
 |  vveeccttoorr
 |      PETSc vector holding the degrees-of-freedom.
 |  
 |  xx
 |      Vector holding the degrees-of-freedom.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  nnaammee
 |      Name of the Function.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from ufl.coefficient.Coefficient:
 |  
 |  ____eeqq____(self, other)
 |      Default comparison of terminals just compare repr strings.
 |  
 |  ____hhaasshh____ = compute_expr_hash(expr)
 |      Compute hashes of *expr* and all its nodes efficiently, without using Python recursion.
 |  
 |  ____rreepprr____(self)
 |      Return string representation this object can be reconstructed from.
 |  
 |  ccoouunntt(self)
 |  
 |  iiss__cceellllwwiissee__ccoonnssttaanntt(self)
 |      Return whether this expression is spatially constant over each cell.
 |  
 |  uuffll__ddoommaaiinn(self)
 |      Shortcut to get the domain of the function space of this coefficient.
 |  
 |  uuffll__ddoommaaiinnss(self)
 |      Return tuple of domains related to this terminal object.
 |  
 |  uuffll__eelleemmeenntt(self)
 |      Shortcut to get the finite element of the function space of this coefficient.
 |  
 |  uuffll__ffuunnccttiioonn__ssppaaccee(self)
 |      Get the function space of this coefficient.
 |  
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from ufl.coefficient.Coefficient:
 |  
 |  uuffll__sshhaappee
 |      Return the associated UFL shape.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from ufl.coefficient.Coefficient:
 |  
 |  ____ddiicctt____
 |      dictionary for instance variables (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from ufl.core.terminal.Terminal:
 |  
 |  eevvaalluuaattee(self, x, mapping, component, index_values, derivatives=())
 |      Get *self* from *mapping* and return the component asked for.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from ufl.core.terminal.Terminal:
 |  
 |  uuffll__ffrreeee__iinnddiicceess = ()
 |  
 |  uuffll__iinnddeexx__ddiimmeennssiioonnss = ()
 |  
 |  uuffll__ooppeerraannddss = ()
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from ufl.core.expr.Expr:
 |  
 |  ____aabbss____ = _abs(self)
 |  
 |  ____aadddd____ = _add(self, o)
 |  
 |  ____bbooooll____(self)
 |      By default, all Expr are nonzero/False.
 |  
 |  ____ccaallll____ = _call(self, arg, mapping=None, component=())
 |  
 |  ____ccoommpplleexx____(self)
 |      Try to evaluate as scalar and cast to complex.
 |  
 |  ____ddiivv____ = _div(self, o)
 |  
 |  ____ffllooaatt____(self)
 |      Try to evaluate as scalar and cast to float.
 |  
 |  ____fflloooorrddiivv____(self, other)
 |      UFL does not support integer division.
 |  
 |  ____ggee____ = _ge(left, right)
 |      UFL operator: A boolean expresion (left >= right) for use with conditional.
 |  
 |  ____ggeettiitteemm____ = _getitem(self, component)
 |  
 |  ____ggeettnneewwaarrggss____(self)
 |      The tuple returned here is passed to as args to cls.__new__(cls, *args).
 |      
 |      This implementation passes the operands, which is () for terminals.
 |      
 |      May be necessary to override if __new__ is implemented in a subclass.
 |  
 |  ____ggtt____ = _gt(left, right)
 |      UFL operator: A boolean expresion (left > right) for use with conditional.
 |  
 |  ____iitteerr____(self)
 |      Iteration over vector expressions.
 |  
 |  ____llee____ = _le(left, right)
 |      UFL operator: A boolean expresion (left <= right) for use with conditional.
 |  
 |  ____lleenn____(self)
 |      Length of expression. Used for iteration over vector expressions.
 |  
 |  ____lltt____ = _lt(left, right)
 |      UFL operator: A boolean expresion (left < right) for use with conditional.
 |  
 |  ____mmuull____ = _mul(self, o)
 |  
 |  ____nnee____ = _ne(self, other)
 |      # != is used at least by tests, possibly in code as well, and must
 |      # mean the opposite of ==, i.e. when evaluated as bool it must mean
 |      # 'not equal representation'.
 |  
 |  ____nneegg____ = _neg(self)
 |      # TODO: Add Negated class for this? Might simplify reductions in Add.
 |  
 |  ____nnoonnzzeerroo____(self)
 |      By default, all Expr are nonzero/False.
 |  
 |  ____ppooss____(self)
 |      Unary + is a no-op.
 |  
 |  ____ppooww____ = _pow