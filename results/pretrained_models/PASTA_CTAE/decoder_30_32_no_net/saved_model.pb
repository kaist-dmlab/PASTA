лы
кЌ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.02v2.6.0-rc2-32-g919f693420e8┴ј

z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:  *
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
: *
dtype0
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

: *
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:*
dtype0
z
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_30/kernel
s
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes

: *
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes
:*
dtype0
z
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

: *
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
:*
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

:*
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
_output_shapes
:*
dtype0
{
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И* 
shared_namedense_31/kernel
t
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes
:	И*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:И*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:И*
dtype0
{
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И* 
shared_namedense_33/kernel
t
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes
:	И*
dtype0
s
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:И*
shared_namedense_33/bias
l
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes	
:И*
dtype0

NoOpNoOp
Х1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ы0
valueу0BС0 BП0
м
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
trainable_variables
regularization_losses
	variables
	keras_api

signatures

_init_input_shape
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
h

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
h

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
h

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
h

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
R
?trainable_variables
@regularization_losses
A	variables
B	keras_api
R
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
R
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
R
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
R
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
R
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
f
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
 
f
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13
Г
trainable_variables
regularization_losses

Wlayers
	variables
Xlayer_regularization_losses
Ymetrics
Zlayer_metrics
[non_trainable_variables
 
 
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
trainable_variables
regularization_losses

\layers
]layer_regularization_losses
	variables
^metrics
_layer_metrics
`non_trainable_variables
[Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Г
trainable_variables
regularization_losses

alayers
blayer_regularization_losses
	variables
cmetrics
dlayer_metrics
enon_trainable_variables
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
Г
#trainable_variables
$regularization_losses

flayers
glayer_regularization_losses
%	variables
hmetrics
ilayer_metrics
jnon_trainable_variables
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1
 

'0
(1
Г
)trainable_variables
*regularization_losses

klayers
llayer_regularization_losses
+	variables
mmetrics
nlayer_metrics
onon_trainable_variables
[Y
VARIABLE_VALUEdense_25/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1
Г
/trainable_variables
0regularization_losses

players
qlayer_regularization_losses
1	variables
rmetrics
slayer_metrics
tnon_trainable_variables
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
Г
5trainable_variables
6regularization_losses

ulayers
vlayer_regularization_losses
7	variables
wmetrics
xlayer_metrics
ynon_trainable_variables
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

90
:1
 

90
:1
Г
;trainable_variables
<regularization_losses

zlayers
{layer_regularization_losses
=	variables
|metrics
}layer_metrics
~non_trainable_variables
 
 
 
▒
?trainable_variables
@regularization_losses

layers
 ђlayer_regularization_losses
A	variables
Ђmetrics
ѓlayer_metrics
Ѓnon_trainable_variables
 
 
 
▓
Ctrainable_variables
Dregularization_losses
ёlayers
 Ёlayer_regularization_losses
E	variables
єmetrics
Єlayer_metrics
ѕnon_trainable_variables
 
 
 
▓
Gtrainable_variables
Hregularization_losses
Ѕlayers
 іlayer_regularization_losses
I	variables
Іmetrics
їlayer_metrics
Їnon_trainable_variables
 
 
 
▓
Ktrainable_variables
Lregularization_losses
јlayers
 Јlayer_regularization_losses
M	variables
љmetrics
Љlayer_metrics
њnon_trainable_variables
 
 
 
▓
Otrainable_variables
Pregularization_losses
Њlayers
 ћlayer_regularization_losses
Q	variables
Ћmetrics
ќlayer_metrics
Ќnon_trainable_variables
 
 
 
▓
Strainable_variables
Tregularization_losses
ўlayers
 Ўlayer_regularization_losses
U	variables
џmetrics
Џlayer_metrics
юnon_trainable_variables
f
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_6Placeholder*'
_output_shapes
:          *
dtype0*
shape:          
з
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6dense_23/kerneldense_23/biasdense_32/kerneldense_32/biasdense_30/kerneldense_30/biasdense_24/kerneldense_24/biasdense_33/kerneldense_33/biasdense_31/kerneldense_31/biasdense_25/kerneldense_25/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ */
f*R(
&__inference_signature_wrapper_10603171
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
д
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__traced_save_10603763
Ѕ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_30/kerneldense_30/biasdense_32/kerneldense_32/biasdense_25/kerneldense_25/biasdense_31/kerneldense_31/biasdense_33/kerneldense_33/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference__traced_restore_10603815▒▓	
Э
ў
+__inference_dense_30_layer_call_fn_10603513

inputs
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_106025772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
»

щ
F__inference_dense_33_layer_call_and_return_conditional_losses_10603582

inputs1
matmul_readvariableop_resource:	И.
biasadd_readvariableop_resource:	И
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         И2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
»

щ
F__inference_dense_33_layer_call_and_return_conditional_losses_10602610

inputs1
matmul_readvariableop_resource:	И.
biasadd_readvariableop_resource:	И
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         И2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ю
d
H__inference_reshape_14_layer_call_and_return_conditional_losses_10603691

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3d
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/4н
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:         2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
ё
э
F__inference_dense_30_layer_call_and_return_conditional_losses_10603504

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ђ?
├
$__inference__traced_restore_10603815
file_prefix2
 assignvariableop_dense_23_kernel:  .
 assignvariableop_1_dense_23_bias: 4
"assignvariableop_2_dense_24_kernel: .
 assignvariableop_3_dense_24_bias:4
"assignvariableop_4_dense_30_kernel: .
 assignvariableop_5_dense_30_bias:4
"assignvariableop_6_dense_32_kernel: .
 assignvariableop_7_dense_32_bias:4
"assignvariableop_8_dense_25_kernel:.
 assignvariableop_9_dense_25_bias:6
#assignvariableop_10_dense_31_kernel:	И0
!assignvariableop_11_dense_31_bias:	И6
#assignvariableop_12_dense_33_kernel:	И0
!assignvariableop_13_dense_33_bias:	И
identity_15ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Е
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*х
valueФBеB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesг
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesШ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЪ
AssignVariableOpAssignVariableOp assignvariableop_dense_23_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_23_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Д
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_24_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ц
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_24_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Д
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_30_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ц
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_30_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Д
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_32_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_32_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Д
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_25_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ц
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_25_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ф
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_31_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Е
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_31_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ф
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_33_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Е
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_33_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpњ
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14f
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_15Щ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ј=
у
E__inference_decoder_layer_call_and_return_conditional_losses_10602966

inputs#
dense_23_10602922:  
dense_23_10602924: #
dense_32_10602927: 
dense_32_10602929:#
dense_30_10602932: 
dense_30_10602934:#
dense_24_10602937: 
dense_24_10602939:$
dense_33_10602942:	И 
dense_33_10602944:	И$
dense_31_10602947:	И 
dense_31_10602949:	И#
dense_25_10602952:
dense_25_10602954:
identity

identity_1

identity_2ѕб dense_23/StatefulPartitionedCallб dense_24/StatefulPartitionedCallб dense_25/StatefulPartitionedCallб dense_30/StatefulPartitionedCallб dense_31/StatefulPartitionedCallб dense_32/StatefulPartitionedCallб dense_33/StatefulPartitionedCallЮ
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_10602922dense_23_10602924*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_106025432"
 dense_23/StatefulPartitionedCall└
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_32_10602927dense_32_10602929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_32_layer_call_and_return_conditional_losses_106025602"
 dense_32/StatefulPartitionedCall└
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_30_10602932dense_30_10602934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_106025772"
 dense_30/StatefulPartitionedCall└
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_10602937dense_24_10602939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_106025942"
 dense_24/StatefulPartitionedCall┴
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_10602942dense_33_10602944*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_33_layer_call_and_return_conditional_losses_106026102"
 dense_33/StatefulPartitionedCall┴
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_10602947dense_31_10602949*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_31_layer_call_and_return_conditional_losses_106026262"
 dense_31/StatefulPartitionedCall└
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_10602952dense_25_10602954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_106026432"
 dense_25/StatefulPartitionedCallЩ
re_lu_1/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_106026632
re_lu_1/PartitionedCallЗ
re_lu/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_106026792
re_lu/PartitionedCallѓ
reshape_10/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_106026932
reshape_10/PartitionedCallЁ
reshape_14/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_14_layer_call_and_return_conditional_losses_106027102
reshape_14/PartitionedCallЃ
reshape_13/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_13_layer_call_and_return_conditional_losses_106027272
reshape_13/PartitionedCallз
softmax/PartitionedCallPartitionedCall#reshape_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_106027342
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityј

Identity_1Identity#reshape_13/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_1ј

Identity_2Identity#reshape_14/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_2├
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ё
э
F__inference_dense_32_layer_call_and_return_conditional_losses_10603524

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Э
ў
+__inference_dense_23_layer_call_fn_10603473

inputs
unknown:  
	unknown_0: 
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_106025432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:          2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
║
ѓ
*__inference_decoder_layer_call_fn_10603453

inputs
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:	И
	unknown_8:	И
	unknown_9:	И

unknown_10:	И

unknown_11:

unknown_12:
identity

identity_1

identity_2ѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_106029662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityІ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*3
_output_shapes!
:         2

Identity_1І

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*3
_output_shapes!
:         2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ё
э
F__inference_dense_24_layer_call_and_return_conditional_losses_10602594

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┴
F
*__inference_softmax_layer_call_fn_10603656

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_106027342
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
й
Ѓ
*__inference_decoder_layer_call_fn_10603038
input_6
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:	И
	unknown_8:	И
	unknown_9:	И

unknown_10:	И

unknown_11:

unknown_12:
identity

identity_1

identity_2ѕбStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_106029662
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityІ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*3
_output_shapes!
:         2

Identity_1І

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*3
_output_shapes!
:         2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_6
ё
э
F__inference_dense_25_layer_call_and_return_conditional_losses_10602643

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ј=
у
E__inference_decoder_layer_call_and_return_conditional_losses_10602739

inputs#
dense_23_10602544:  
dense_23_10602546: #
dense_32_10602561: 
dense_32_10602563:#
dense_30_10602578: 
dense_30_10602580:#
dense_24_10602595: 
dense_24_10602597:$
dense_33_10602611:	И 
dense_33_10602613:	И$
dense_31_10602627:	И 
dense_31_10602629:	И#
dense_25_10602644:
dense_25_10602646:
identity

identity_1

identity_2ѕб dense_23/StatefulPartitionedCallб dense_24/StatefulPartitionedCallб dense_25/StatefulPartitionedCallб dense_30/StatefulPartitionedCallб dense_31/StatefulPartitionedCallб dense_32/StatefulPartitionedCallб dense_33/StatefulPartitionedCallЮ
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_10602544dense_23_10602546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_106025432"
 dense_23/StatefulPartitionedCall└
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_32_10602561dense_32_10602563*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_32_layer_call_and_return_conditional_losses_106025602"
 dense_32/StatefulPartitionedCall└
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_30_10602578dense_30_10602580*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_106025772"
 dense_30/StatefulPartitionedCall└
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_10602595dense_24_10602597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_106025942"
 dense_24/StatefulPartitionedCall┴
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_10602611dense_33_10602613*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_33_layer_call_and_return_conditional_losses_106026102"
 dense_33/StatefulPartitionedCall┴
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_10602627dense_31_10602629*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_31_layer_call_and_return_conditional_losses_106026262"
 dense_31/StatefulPartitionedCall└
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_10602644dense_25_10602646*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_106026432"
 dense_25/StatefulPartitionedCallЩ
re_lu_1/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_106026632
re_lu_1/PartitionedCallЗ
re_lu/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_106026792
re_lu/PartitionedCallѓ
reshape_10/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_106026932
reshape_10/PartitionedCallЁ
reshape_14/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_14_layer_call_and_return_conditional_losses_106027102
reshape_14/PartitionedCallЃ
reshape_13/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_13_layer_call_and_return_conditional_losses_106027272
reshape_13/PartitionedCallз
softmax/PartitionedCallPartitionedCall#reshape_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_106027342
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityј

Identity_1Identity#reshape_13/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_1ј

Identity_2Identity#reshape_14/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_2├
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ь

_
C__inference_re_lu_layer_call_and_return_conditional_losses_10602679

inputs
identityL
NegNeginputs*
T0*(
_output_shapes
:         И2
NegP
ReluReluNeg:y:0*
T0*(
_output_shapes
:         И2
ReluS
Relu_1Reluinputs*
T0*(
_output_shapes
:         И2
Relu_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1њ
clip_by_value/MinimumMinimumRelu_1:activations:0Const:output:0*
T0*(
_output_shapes
:         И2
clip_by_value/MinimumЅ
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*(
_output_shapes
:         И2
clip_by_valueW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2	
Const_2j
mulMulConst_2:output:0Relu:activations:0*
T0*(
_output_shapes
:         И2
mul`
subSubclip_by_value:z:0mul:z:0*
T0*(
_output_shapes
:         И2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:         И2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
ш
ѓ
E__inference_decoder_layer_call_and_return_conditional_losses_10603275

inputs9
'dense_23_matmul_readvariableop_resource:  6
(dense_23_biasadd_readvariableop_resource: 9
'dense_32_matmul_readvariableop_resource: 6
(dense_32_biasadd_readvariableop_resource:9
'dense_30_matmul_readvariableop_resource: 6
(dense_30_biasadd_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource::
'dense_33_matmul_readvariableop_resource:	И7
(dense_33_biasadd_readvariableop_resource:	И:
'dense_31_matmul_readvariableop_resource:	И7
(dense_31_biasadd_readvariableop_resource:	И9
'dense_25_matmul_readvariableop_resource:6
(dense_25_biasadd_readvariableop_resource:
identity

identity_1

identity_2ѕбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpбdense_24/BiasAdd/ReadVariableOpбdense_24/MatMul/ReadVariableOpбdense_25/BiasAdd/ReadVariableOpбdense_25/MatMul/ReadVariableOpбdense_30/BiasAdd/ReadVariableOpбdense_30/MatMul/ReadVariableOpбdense_31/BiasAdd/ReadVariableOpбdense_31/MatMul/ReadVariableOpбdense_32/BiasAdd/ReadVariableOpбdense_32/MatMul/ReadVariableOpбdense_33/BiasAdd/ReadVariableOpбdense_33/MatMul/ReadVariableOpе
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_23/MatMul/ReadVariableOpј
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_23/BiasAdd/ReadVariableOpЦ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_23/BiasAdds
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_23/Reluе
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_32/MatMul/ReadVariableOpБ
dense_32/MatMulMatMuldense_23/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_32/MatMulД
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOpЦ
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_32/BiasAdds
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_32/Reluе
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_30/MatMul/ReadVariableOpБ
dense_30/MatMulMatMuldense_23/Relu:activations:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_30/MatMulД
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_30/BiasAdd/ReadVariableOpЦ
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_30/Reluе
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOpБ
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/MatMulД
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpЦ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/BiasAdds
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_24/ReluЕ
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype02 
dense_33/MatMul/ReadVariableOpц
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
dense_33/MatMulе
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02!
dense_33/BiasAdd/ReadVariableOpд
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
dense_33/BiasAddЕ
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype02 
dense_31/MatMul/ReadVariableOpц
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
dense_31/MatMulе
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02!
dense_31/BiasAdd/ReadVariableOpд
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
dense_31/BiasAddе
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_25/MatMul/ReadVariableOpБ
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_25/MatMulД
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOpЦ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_25/Reluo
re_lu_1/NegNegdense_33/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
re_lu_1/Negh
re_lu_1/ReluRelure_lu_1/Neg:y:0*
T0*(
_output_shapes
:         И2
re_lu_1/Reluv
re_lu_1/Relu_1Reludense_33/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
re_lu_1/Relu_1c
re_lu_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
re_lu_1/Constg
re_lu_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
re_lu_1/Const_1▓
re_lu_1/clip_by_value/MinimumMinimumre_lu_1/Relu_1:activations:0re_lu_1/Const:output:0*
T0*(
_output_shapes
:         И2
re_lu_1/clip_by_value/MinimumЕ
re_lu_1/clip_by_valueMaximum!re_lu_1/clip_by_value/Minimum:z:0re_lu_1/Const_1:output:0*
T0*(
_output_shapes
:         И2
re_lu_1/clip_by_valueg
re_lu_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
re_lu_1/Const_2і
re_lu_1/mulMulre_lu_1/Const_2:output:0re_lu_1/Relu:activations:0*
T0*(
_output_shapes
:         И2
re_lu_1/mulђ
re_lu_1/subSubre_lu_1/clip_by_value:z:0re_lu_1/mul:z:0*
T0*(
_output_shapes
:         И2
re_lu_1/subk
	re_lu/NegNegdense_31/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
	re_lu/Negb

re_lu/ReluRelure_lu/Neg:y:0*
T0*(
_output_shapes
:         И2

re_lu/Relur
re_lu/Relu_1Reludense_31/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
re_lu/Relu_1_
re_lu/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
re_lu/Constc
re_lu/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
re_lu/Const_1ф
re_lu/clip_by_value/MinimumMinimumre_lu/Relu_1:activations:0re_lu/Const:output:0*
T0*(
_output_shapes
:         И2
re_lu/clip_by_value/MinimumА
re_lu/clip_by_valueMaximumre_lu/clip_by_value/Minimum:z:0re_lu/Const_1:output:0*
T0*(
_output_shapes
:         И2
re_lu/clip_by_valuec
re_lu/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
re_lu/Const_2ѓ
	re_lu/mulMulre_lu/Const_2:output:0re_lu/Relu:activations:0*
T0*(
_output_shapes
:         И2
	re_lu/mulx
	re_lu/subSubre_lu/clip_by_value:z:0re_lu/mul:z:0*
T0*(
_output_shapes
:         И2
	re_lu/subo
reshape_10/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
reshape_10/Shapeі
reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_10/strided_slice/stackј
 reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_1ј
 reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_2ц
reshape_10/strided_sliceStridedSlicereshape_10/Shape:output:0'reshape_10/strided_slice/stack:output:0)reshape_10/strided_slice/stack_1:output:0)reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_10/strided_slicez
reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/1▓
reshape_10/Reshape/shapePack!reshape_10/strided_slice:output:0#reshape_10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_10/Reshape/shapeЦ
reshape_10/ReshapeReshapedense_25/Relu:activations:0!reshape_10/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
reshape_10/Reshapec
reshape_14/ShapeShapere_lu_1/sub:z:0*
T0*
_output_shapes
:2
reshape_14/Shapeі
reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_14/strided_slice/stackј
 reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_1ј
 reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_2ц
reshape_14/strided_sliceStridedSlicereshape_14/Shape:output:0'reshape_14/strided_slice/stack:output:0)reshape_14/strided_slice/stack_1:output:0)reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_14/strided_slicez
reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/1z
reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/2z
reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/3z
reshape_14/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/4А
reshape_14/Reshape/shapePack!reshape_14/strided_slice:output:0#reshape_14/Reshape/shape/1:output:0#reshape_14/Reshape/shape/2:output:0#reshape_14/Reshape/shape/3:output:0#reshape_14/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape_14/Reshape/shapeЦ
reshape_14/ReshapeReshapere_lu_1/sub:z:0!reshape_14/Reshape/shape:output:0*
T0*3
_output_shapes!
:         2
reshape_14/Reshapea
reshape_13/ShapeShapere_lu/sub:z:0*
T0*
_output_shapes
:2
reshape_13/Shapeі
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stackј
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1ј
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2ц
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/1z
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/2z
reshape_13/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/3z
reshape_13/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/4А
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0#reshape_13/Reshape/shape/3:output:0#reshape_13/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shapeБ
reshape_13/ReshapeReshapere_lu/sub:z:0!reshape_13/Reshape/shape:output:0*
T0*3
_output_shapes!
:         2
reshape_13/Reshape|
softmax/SoftmaxSoftmaxreshape_10/Reshape:output:0*
T0*'
_output_shapes
:         2
softmax/Softmaxt
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

Identityє

Identity_1Identityreshape_13/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_1є

Identity_2Identityreshape_14/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_2Б
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ю
d
H__inference_reshape_13_layer_call_and_return_conditional_losses_10603671

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3d
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/4н
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:         2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
Э
ў
+__inference_dense_32_layer_call_fn_10603533

inputs
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_32_layer_call_and_return_conditional_losses_106025602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
­

a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_10603641

inputs
identityL
NegNeginputs*
T0*(
_output_shapes
:         И2
NegP
ReluReluNeg:y:0*
T0*(
_output_shapes
:         И2
ReluS
Relu_1Reluinputs*
T0*(
_output_shapes
:         И2
Relu_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1њ
clip_by_value/MinimumMinimumRelu_1:activations:0Const:output:0*
T0*(
_output_shapes
:         И2
clip_by_value/MinimumЅ
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*(
_output_shapes
:         И2
clip_by_valueW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2	
Const_2j
mulMulConst_2:output:0Relu:activations:0*
T0*(
_output_shapes
:         И2
mul`
subSubclip_by_value:z:0mul:z:0*
T0*(
_output_shapes
:         И2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:         И2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
ё
э
F__inference_dense_30_layer_call_and_return_conditional_losses_10602577

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ё
э
F__inference_dense_23_layer_call_and_return_conditional_losses_10603464

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ч
џ
+__inference_dense_33_layer_call_fn_10603591

inputs
unknown:	И
	unknown_0:	И
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_33_layer_call_and_return_conditional_losses_106026102
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         И2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Љ=
У
E__inference_decoder_layer_call_and_return_conditional_losses_10603085
input_6#
dense_23_10603041:  
dense_23_10603043: #
dense_32_10603046: 
dense_32_10603048:#
dense_30_10603051: 
dense_30_10603053:#
dense_24_10603056: 
dense_24_10603058:$
dense_33_10603061:	И 
dense_33_10603063:	И$
dense_31_10603066:	И 
dense_31_10603068:	И#
dense_25_10603071:
dense_25_10603073:
identity

identity_1

identity_2ѕб dense_23/StatefulPartitionedCallб dense_24/StatefulPartitionedCallб dense_25/StatefulPartitionedCallб dense_30/StatefulPartitionedCallб dense_31/StatefulPartitionedCallб dense_32/StatefulPartitionedCallб dense_33/StatefulPartitionedCallъ
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_23_10603041dense_23_10603043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_106025432"
 dense_23/StatefulPartitionedCall└
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_32_10603046dense_32_10603048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_32_layer_call_and_return_conditional_losses_106025602"
 dense_32/StatefulPartitionedCall└
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_30_10603051dense_30_10603053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_106025772"
 dense_30/StatefulPartitionedCall└
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_10603056dense_24_10603058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_106025942"
 dense_24/StatefulPartitionedCall┴
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_10603061dense_33_10603063*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_33_layer_call_and_return_conditional_losses_106026102"
 dense_33/StatefulPartitionedCall┴
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_10603066dense_31_10603068*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_31_layer_call_and_return_conditional_losses_106026262"
 dense_31/StatefulPartitionedCall└
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_10603071dense_25_10603073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_106026432"
 dense_25/StatefulPartitionedCallЩ
re_lu_1/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_106026632
re_lu_1/PartitionedCallЗ
re_lu/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_106026792
re_lu/PartitionedCallѓ
reshape_10/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_106026932
reshape_10/PartitionedCallЁ
reshape_14/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_14_layer_call_and_return_conditional_losses_106027102
reshape_14/PartitionedCallЃ
reshape_13/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_13_layer_call_and_return_conditional_losses_106027272
reshape_13/PartitionedCallз
softmax/PartitionedCallPartitionedCall#reshape_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_106027342
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityј

Identity_1Identity#reshape_13/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_1ј

Identity_2Identity#reshape_14/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_2├
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_6
ё
э
F__inference_dense_24_layer_call_and_return_conditional_losses_10603484

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ё
э
F__inference_dense_32_layer_call_and_return_conditional_losses_10602560

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ѓ
d
H__inference_reshape_10_layer_call_and_return_conditional_losses_10603603

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1є
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:         2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ќ
 
&__inference_signature_wrapper_10603171
input_6
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:	И
	unknown_8:	И
	unknown_9:	И

unknown_10:	И

unknown_11:

unknown_12:
identity

identity_1

identity_2ѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference__wrapped_model_106025252
StatefulPartitionedCallЄ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

IdentityІ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*3
_output_shapes!
:         2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_6
Љ=
У
E__inference_decoder_layer_call_and_return_conditional_losses_10603132
input_6#
dense_23_10603088:  
dense_23_10603090: #
dense_32_10603093: 
dense_32_10603095:#
dense_30_10603098: 
dense_30_10603100:#
dense_24_10603103: 
dense_24_10603105:$
dense_33_10603108:	И 
dense_33_10603110:	И$
dense_31_10603113:	И 
dense_31_10603115:	И#
dense_25_10603118:
dense_25_10603120:
identity

identity_1

identity_2ѕб dense_23/StatefulPartitionedCallб dense_24/StatefulPartitionedCallб dense_25/StatefulPartitionedCallб dense_30/StatefulPartitionedCallб dense_31/StatefulPartitionedCallб dense_32/StatefulPartitionedCallб dense_33/StatefulPartitionedCallъ
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_23_10603088dense_23_10603090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_106025432"
 dense_23/StatefulPartitionedCall└
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_32_10603093dense_32_10603095*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_32_layer_call_and_return_conditional_losses_106025602"
 dense_32/StatefulPartitionedCall└
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_30_10603098dense_30_10603100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_30_layer_call_and_return_conditional_losses_106025772"
 dense_30/StatefulPartitionedCall└
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_10603103dense_24_10603105*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_106025942"
 dense_24/StatefulPartitionedCall┴
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_10603108dense_33_10603110*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_33_layer_call_and_return_conditional_losses_106026102"
 dense_33/StatefulPartitionedCall┴
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_10603113dense_31_10603115*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_31_layer_call_and_return_conditional_losses_106026262"
 dense_31/StatefulPartitionedCall└
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_10603118dense_25_10603120*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_106026432"
 dense_25/StatefulPartitionedCallЩ
re_lu_1/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_106026632
re_lu_1/PartitionedCallЗ
re_lu/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_106026792
re_lu/PartitionedCallѓ
reshape_10/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_106026932
reshape_10/PartitionedCallЁ
reshape_14/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_14_layer_call_and_return_conditional_losses_106027102
reshape_14/PartitionedCallЃ
reshape_13/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_13_layer_call_and_return_conditional_losses_106027272
reshape_13/PartitionedCallз
softmax/PartitionedCallPartitionedCall#reshape_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_106027342
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityј

Identity_1Identity#reshape_13/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_1ј

Identity_2Identity#reshape_14/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_2├
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_6
Ѓ
d
H__inference_reshape_10_layer_call_and_return_conditional_losses_10602693

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1є
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:         2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ь

_
C__inference_re_lu_layer_call_and_return_conditional_losses_10603622

inputs
identityL
NegNeginputs*
T0*(
_output_shapes
:         И2
NegP
ReluReluNeg:y:0*
T0*(
_output_shapes
:         И2
ReluS
Relu_1Reluinputs*
T0*(
_output_shapes
:         И2
Relu_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1њ
clip_by_value/MinimumMinimumRelu_1:activations:0Const:output:0*
T0*(
_output_shapes
:         И2
clip_by_value/MinimumЅ
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*(
_output_shapes
:         И2
clip_by_valueW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2	
Const_2j
mulMulConst_2:output:0Relu:activations:0*
T0*(
_output_shapes
:         И2
mul`
subSubclip_by_value:z:0mul:z:0*
T0*(
_output_shapes
:         И2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:         И2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
р
I
-__inference_reshape_14_layer_call_fn_10603696

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_14_layer_call_and_return_conditional_losses_106027102
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
ё
э
F__inference_dense_25_layer_call_and_return_conditional_losses_10603544

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
т
a
E__inference_softmax_layer_call_and_return_conditional_losses_10603651

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Э
ў
+__inference_dense_25_layer_call_fn_10603553

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_106026432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
»

щ
F__inference_dense_31_layer_call_and_return_conditional_losses_10602626

inputs1
matmul_readvariableop_resource:	И.
biasadd_readvariableop_resource:	И
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         И2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ё
э
F__inference_dense_23_layer_call_and_return_conditional_losses_10602543

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:          2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
»

щ
F__inference_dense_31_layer_call_and_return_conditional_losses_10603563

inputs1
matmul_readvariableop_resource:	И.
biasadd_readvariableop_resource:	И
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:         И2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ш
ѓ
E__inference_decoder_layer_call_and_return_conditional_losses_10603379

inputs9
'dense_23_matmul_readvariableop_resource:  6
(dense_23_biasadd_readvariableop_resource: 9
'dense_32_matmul_readvariableop_resource: 6
(dense_32_biasadd_readvariableop_resource:9
'dense_30_matmul_readvariableop_resource: 6
(dense_30_biasadd_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource::
'dense_33_matmul_readvariableop_resource:	И7
(dense_33_biasadd_readvariableop_resource:	И:
'dense_31_matmul_readvariableop_resource:	И7
(dense_31_biasadd_readvariableop_resource:	И9
'dense_25_matmul_readvariableop_resource:6
(dense_25_biasadd_readvariableop_resource:
identity

identity_1

identity_2ѕбdense_23/BiasAdd/ReadVariableOpбdense_23/MatMul/ReadVariableOpбdense_24/BiasAdd/ReadVariableOpбdense_24/MatMul/ReadVariableOpбdense_25/BiasAdd/ReadVariableOpбdense_25/MatMul/ReadVariableOpбdense_30/BiasAdd/ReadVariableOpбdense_30/MatMul/ReadVariableOpбdense_31/BiasAdd/ReadVariableOpбdense_31/MatMul/ReadVariableOpбdense_32/BiasAdd/ReadVariableOpбdense_32/MatMul/ReadVariableOpбdense_33/BiasAdd/ReadVariableOpбdense_33/MatMul/ReadVariableOpе
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_23/MatMul/ReadVariableOpј
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_23/MatMulД
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_23/BiasAdd/ReadVariableOpЦ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_23/BiasAdds
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_23/Reluе
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_32/MatMul/ReadVariableOpБ
dense_32/MatMulMatMuldense_23/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_32/MatMulД
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOpЦ
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_32/BiasAdds
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_32/Reluе
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_30/MatMul/ReadVariableOpБ
dense_30/MatMulMatMuldense_23/Relu:activations:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_30/MatMulД
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_30/BiasAdd/ReadVariableOpЦ
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_30/Reluе
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOpБ
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/MatMulД
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpЦ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/BiasAdds
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_24/ReluЕ
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype02 
dense_33/MatMul/ReadVariableOpц
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
dense_33/MatMulе
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02!
dense_33/BiasAdd/ReadVariableOpд
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
dense_33/BiasAddЕ
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype02 
dense_31/MatMul/ReadVariableOpц
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
dense_31/MatMulе
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02!
dense_31/BiasAdd/ReadVariableOpд
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
dense_31/BiasAddе
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_25/MatMul/ReadVariableOpБ
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_25/MatMulД
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOpЦ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_25/Reluo
re_lu_1/NegNegdense_33/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
re_lu_1/Negh
re_lu_1/ReluRelure_lu_1/Neg:y:0*
T0*(
_output_shapes
:         И2
re_lu_1/Reluv
re_lu_1/Relu_1Reludense_33/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
re_lu_1/Relu_1c
re_lu_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
re_lu_1/Constg
re_lu_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
re_lu_1/Const_1▓
re_lu_1/clip_by_value/MinimumMinimumre_lu_1/Relu_1:activations:0re_lu_1/Const:output:0*
T0*(
_output_shapes
:         И2
re_lu_1/clip_by_value/MinimumЕ
re_lu_1/clip_by_valueMaximum!re_lu_1/clip_by_value/Minimum:z:0re_lu_1/Const_1:output:0*
T0*(
_output_shapes
:         И2
re_lu_1/clip_by_valueg
re_lu_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
re_lu_1/Const_2і
re_lu_1/mulMulre_lu_1/Const_2:output:0re_lu_1/Relu:activations:0*
T0*(
_output_shapes
:         И2
re_lu_1/mulђ
re_lu_1/subSubre_lu_1/clip_by_value:z:0re_lu_1/mul:z:0*
T0*(
_output_shapes
:         И2
re_lu_1/subk
	re_lu/NegNegdense_31/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
	re_lu/Negb

re_lu/ReluRelure_lu/Neg:y:0*
T0*(
_output_shapes
:         И2

re_lu/Relur
re_lu/Relu_1Reludense_31/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
re_lu/Relu_1_
re_lu/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
re_lu/Constc
re_lu/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
re_lu/Const_1ф
re_lu/clip_by_value/MinimumMinimumre_lu/Relu_1:activations:0re_lu/Const:output:0*
T0*(
_output_shapes
:         И2
re_lu/clip_by_value/MinimumА
re_lu/clip_by_valueMaximumre_lu/clip_by_value/Minimum:z:0re_lu/Const_1:output:0*
T0*(
_output_shapes
:         И2
re_lu/clip_by_valuec
re_lu/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
re_lu/Const_2ѓ
	re_lu/mulMulre_lu/Const_2:output:0re_lu/Relu:activations:0*
T0*(
_output_shapes
:         И2
	re_lu/mulx
	re_lu/subSubre_lu/clip_by_value:z:0re_lu/mul:z:0*
T0*(
_output_shapes
:         И2
	re_lu/subo
reshape_10/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
reshape_10/Shapeі
reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_10/strided_slice/stackј
 reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_1ј
 reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_2ц
reshape_10/strided_sliceStridedSlicereshape_10/Shape:output:0'reshape_10/strided_slice/stack:output:0)reshape_10/strided_slice/stack_1:output:0)reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_10/strided_slicez
reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_10/Reshape/shape/1▓
reshape_10/Reshape/shapePack!reshape_10/strided_slice:output:0#reshape_10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_10/Reshape/shapeЦ
reshape_10/ReshapeReshapedense_25/Relu:activations:0!reshape_10/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
reshape_10/Reshapec
reshape_14/ShapeShapere_lu_1/sub:z:0*
T0*
_output_shapes
:2
reshape_14/Shapeі
reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_14/strided_slice/stackј
 reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_1ј
 reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_2ц
reshape_14/strided_sliceStridedSlicereshape_14/Shape:output:0'reshape_14/strided_slice/stack:output:0)reshape_14/strided_slice/stack_1:output:0)reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_14/strided_slicez
reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/1z
reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/2z
reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/3z
reshape_14/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_14/Reshape/shape/4А
reshape_14/Reshape/shapePack!reshape_14/strided_slice:output:0#reshape_14/Reshape/shape/1:output:0#reshape_14/Reshape/shape/2:output:0#reshape_14/Reshape/shape/3:output:0#reshape_14/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape_14/Reshape/shapeЦ
reshape_14/ReshapeReshapere_lu_1/sub:z:0!reshape_14/Reshape/shape:output:0*
T0*3
_output_shapes!
:         2
reshape_14/Reshapea
reshape_13/ShapeShapere_lu/sub:z:0*
T0*
_output_shapes
:2
reshape_13/Shapeі
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stackј
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1ј
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2ц
reshape_13/strided_sliceStridedSlicereshape_13/Shape:output:0'reshape_13/strided_slice/stack:output:0)reshape_13/strided_slice/stack_1:output:0)reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_13/strided_slicez
reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/1z
reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/2z
reshape_13/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/3z
reshape_13/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_13/Reshape/shape/4А
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0#reshape_13/Reshape/shape/3:output:0#reshape_13/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shapeБ
reshape_13/ReshapeReshapere_lu/sub:z:0!reshape_13/Reshape/shape:output:0*
T0*3
_output_shapes!
:         2
reshape_13/Reshape|
softmax/SoftmaxSoftmaxreshape_10/Reshape:output:0*
T0*'
_output_shapes
:         2
softmax/Softmaxt
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

Identityє

Identity_1Identityreshape_13/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_1є

Identity_2Identityreshape_14/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_2Б
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2B
dense_32/BiasAdd/ReadVariableOpdense_32/BiasAdd/ReadVariableOp2@
dense_32/MatMul/ReadVariableOpdense_32/MatMul/ReadVariableOp2B
dense_33/BiasAdd/ReadVariableOpdense_33/BiasAdd/ReadVariableOp2@
dense_33/MatMul/ReadVariableOpdense_33/MatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ч
џ
+__inference_dense_31_layer_call_fn_10603572

inputs
unknown:	И
	unknown_0:	И
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_31_layer_call_and_return_conditional_losses_106026262
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         И2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╗(
ђ
!__inference__traced_save_10603763
file_prefix.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameБ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*х
valueФBеB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesд
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesб
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*І
_input_shapesz
x: :  : : :: :: ::::	И:И:	И:И: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::%!

_output_shapes
:	И:!

_output_shapes	
:И:%!

_output_shapes
:	И:!

_output_shapes	
:И:

_output_shapes
: 
­

a
E__inference_re_lu_1_layer_call_and_return_conditional_losses_10602663

inputs
identityL
NegNeginputs*
T0*(
_output_shapes
:         И2
NegP
ReluReluNeg:y:0*
T0*(
_output_shapes
:         И2
ReluS
Relu_1Reluinputs*
T0*(
_output_shapes
:         И2
Relu_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1њ
clip_by_value/MinimumMinimumRelu_1:activations:0Const:output:0*
T0*(
_output_shapes
:         И2
clip_by_value/MinimumЅ
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*(
_output_shapes
:         И2
clip_by_valueW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2	
Const_2j
mulMulConst_2:output:0Relu:activations:0*
T0*(
_output_shapes
:         И2
mul`
subSubclip_by_value:z:0mul:z:0*
T0*(
_output_shapes
:         И2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:         И2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
Э
ў
+__inference_dense_24_layer_call_fn_10603493

inputs
unknown: 
	unknown_0:
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_106025942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ЦЋ
┴
#__inference__wrapped_model_10602525
input_6A
/decoder_dense_23_matmul_readvariableop_resource:  >
0decoder_dense_23_biasadd_readvariableop_resource: A
/decoder_dense_32_matmul_readvariableop_resource: >
0decoder_dense_32_biasadd_readvariableop_resource:A
/decoder_dense_30_matmul_readvariableop_resource: >
0decoder_dense_30_biasadd_readvariableop_resource:A
/decoder_dense_24_matmul_readvariableop_resource: >
0decoder_dense_24_biasadd_readvariableop_resource:B
/decoder_dense_33_matmul_readvariableop_resource:	И?
0decoder_dense_33_biasadd_readvariableop_resource:	ИB
/decoder_dense_31_matmul_readvariableop_resource:	И?
0decoder_dense_31_biasadd_readvariableop_resource:	ИA
/decoder_dense_25_matmul_readvariableop_resource:>
0decoder_dense_25_biasadd_readvariableop_resource:
identity

identity_1

identity_2ѕб'decoder/dense_23/BiasAdd/ReadVariableOpб&decoder/dense_23/MatMul/ReadVariableOpб'decoder/dense_24/BiasAdd/ReadVariableOpб&decoder/dense_24/MatMul/ReadVariableOpб'decoder/dense_25/BiasAdd/ReadVariableOpб&decoder/dense_25/MatMul/ReadVariableOpб'decoder/dense_30/BiasAdd/ReadVariableOpб&decoder/dense_30/MatMul/ReadVariableOpб'decoder/dense_31/BiasAdd/ReadVariableOpб&decoder/dense_31/MatMul/ReadVariableOpб'decoder/dense_32/BiasAdd/ReadVariableOpб&decoder/dense_32/MatMul/ReadVariableOpб'decoder/dense_33/BiasAdd/ReadVariableOpб&decoder/dense_33/MatMul/ReadVariableOp└
&decoder/dense_23/MatMul/ReadVariableOpReadVariableOp/decoder_dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02(
&decoder/dense_23/MatMul/ReadVariableOpД
decoder/dense_23/MatMulMatMulinput_6.decoder/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
decoder/dense_23/MatMul┐
'decoder/dense_23/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'decoder/dense_23/BiasAdd/ReadVariableOp┼
decoder/dense_23/BiasAddBiasAdd!decoder/dense_23/MatMul:product:0/decoder/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
decoder/dense_23/BiasAddІ
decoder/dense_23/ReluRelu!decoder/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:          2
decoder/dense_23/Relu└
&decoder/dense_32/MatMul/ReadVariableOpReadVariableOp/decoder_dense_32_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_32/MatMul/ReadVariableOp├
decoder/dense_32/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
decoder/dense_32/MatMul┐
'decoder/dense_32/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_32/BiasAdd/ReadVariableOp┼
decoder/dense_32/BiasAddBiasAdd!decoder/dense_32/MatMul:product:0/decoder/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
decoder/dense_32/BiasAddІ
decoder/dense_32/ReluRelu!decoder/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:         2
decoder/dense_32/Relu└
&decoder/dense_30/MatMul/ReadVariableOpReadVariableOp/decoder_dense_30_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_30/MatMul/ReadVariableOp├
decoder/dense_30/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
decoder/dense_30/MatMul┐
'decoder/dense_30/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_30/BiasAdd/ReadVariableOp┼
decoder/dense_30/BiasAddBiasAdd!decoder/dense_30/MatMul:product:0/decoder/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
decoder/dense_30/BiasAddІ
decoder/dense_30/ReluRelu!decoder/dense_30/BiasAdd:output:0*
T0*'
_output_shapes
:         2
decoder/dense_30/Relu└
&decoder/dense_24/MatMul/ReadVariableOpReadVariableOp/decoder_dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_24/MatMul/ReadVariableOp├
decoder/dense_24/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
decoder/dense_24/MatMul┐
'decoder/dense_24/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_24/BiasAdd/ReadVariableOp┼
decoder/dense_24/BiasAddBiasAdd!decoder/dense_24/MatMul:product:0/decoder/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
decoder/dense_24/BiasAddІ
decoder/dense_24/ReluRelu!decoder/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2
decoder/dense_24/Relu┴
&decoder/dense_33/MatMul/ReadVariableOpReadVariableOp/decoder_dense_33_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype02(
&decoder/dense_33/MatMul/ReadVariableOp─
decoder/dense_33/MatMulMatMul#decoder/dense_32/Relu:activations:0.decoder/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
decoder/dense_33/MatMul└
'decoder/dense_33/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02)
'decoder/dense_33/BiasAdd/ReadVariableOpк
decoder/dense_33/BiasAddBiasAdd!decoder/dense_33/MatMul:product:0/decoder/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
decoder/dense_33/BiasAdd┴
&decoder/dense_31/MatMul/ReadVariableOpReadVariableOp/decoder_dense_31_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype02(
&decoder/dense_31/MatMul/ReadVariableOp─
decoder/dense_31/MatMulMatMul#decoder/dense_30/Relu:activations:0.decoder/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
decoder/dense_31/MatMul└
'decoder/dense_31/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02)
'decoder/dense_31/BiasAdd/ReadVariableOpк
decoder/dense_31/BiasAddBiasAdd!decoder/dense_31/MatMul:product:0/decoder/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
decoder/dense_31/BiasAdd└
&decoder/dense_25/MatMul/ReadVariableOpReadVariableOp/decoder_dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&decoder/dense_25/MatMul/ReadVariableOp├
decoder/dense_25/MatMulMatMul#decoder/dense_24/Relu:activations:0.decoder/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
decoder/dense_25/MatMul┐
'decoder/dense_25/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_25/BiasAdd/ReadVariableOp┼
decoder/dense_25/BiasAddBiasAdd!decoder/dense_25/MatMul:product:0/decoder/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
decoder/dense_25/BiasAddІ
decoder/dense_25/ReluRelu!decoder/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:         2
decoder/dense_25/ReluЄ
decoder/re_lu_1/NegNeg!decoder/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
decoder/re_lu_1/Negђ
decoder/re_lu_1/ReluReludecoder/re_lu_1/Neg:y:0*
T0*(
_output_shapes
:         И2
decoder/re_lu_1/Reluј
decoder/re_lu_1/Relu_1Relu!decoder/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
decoder/re_lu_1/Relu_1s
decoder/re_lu_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
decoder/re_lu_1/Constw
decoder/re_lu_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
decoder/re_lu_1/Const_1м
%decoder/re_lu_1/clip_by_value/MinimumMinimum$decoder/re_lu_1/Relu_1:activations:0decoder/re_lu_1/Const:output:0*
T0*(
_output_shapes
:         И2'
%decoder/re_lu_1/clip_by_value/Minimum╔
decoder/re_lu_1/clip_by_valueMaximum)decoder/re_lu_1/clip_by_value/Minimum:z:0 decoder/re_lu_1/Const_1:output:0*
T0*(
_output_shapes
:         И2
decoder/re_lu_1/clip_by_valuew
decoder/re_lu_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
decoder/re_lu_1/Const_2ф
decoder/re_lu_1/mulMul decoder/re_lu_1/Const_2:output:0"decoder/re_lu_1/Relu:activations:0*
T0*(
_output_shapes
:         И2
decoder/re_lu_1/mulа
decoder/re_lu_1/subSub!decoder/re_lu_1/clip_by_value:z:0decoder/re_lu_1/mul:z:0*
T0*(
_output_shapes
:         И2
decoder/re_lu_1/subЃ
decoder/re_lu/NegNeg!decoder/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
decoder/re_lu/Negz
decoder/re_lu/ReluReludecoder/re_lu/Neg:y:0*
T0*(
_output_shapes
:         И2
decoder/re_lu/Reluі
decoder/re_lu/Relu_1Relu!decoder/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:         И2
decoder/re_lu/Relu_1o
decoder/re_lu/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
decoder/re_lu/Consts
decoder/re_lu/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
decoder/re_lu/Const_1╩
#decoder/re_lu/clip_by_value/MinimumMinimum"decoder/re_lu/Relu_1:activations:0decoder/re_lu/Const:output:0*
T0*(
_output_shapes
:         И2%
#decoder/re_lu/clip_by_value/Minimum┴
decoder/re_lu/clip_by_valueMaximum'decoder/re_lu/clip_by_value/Minimum:z:0decoder/re_lu/Const_1:output:0*
T0*(
_output_shapes
:         И2
decoder/re_lu/clip_by_values
decoder/re_lu/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
decoder/re_lu/Const_2б
decoder/re_lu/mulMuldecoder/re_lu/Const_2:output:0 decoder/re_lu/Relu:activations:0*
T0*(
_output_shapes
:         И2
decoder/re_lu/mulў
decoder/re_lu/subSubdecoder/re_lu/clip_by_value:z:0decoder/re_lu/mul:z:0*
T0*(
_output_shapes
:         И2
decoder/re_lu/subЄ
decoder/reshape_10/ShapeShape#decoder/dense_25/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_10/Shapeџ
&decoder/reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_10/strided_slice/stackъ
(decoder/reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_10/strided_slice/stack_1ъ
(decoder/reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_10/strided_slice/stack_2н
 decoder/reshape_10/strided_sliceStridedSlice!decoder/reshape_10/Shape:output:0/decoder/reshape_10/strided_slice/stack:output:01decoder/reshape_10/strided_slice/stack_1:output:01decoder/reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_10/strided_sliceі
"decoder/reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_10/Reshape/shape/1м
 decoder/reshape_10/Reshape/shapePack)decoder/reshape_10/strided_slice:output:0+decoder/reshape_10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_10/Reshape/shape┼
decoder/reshape_10/ReshapeReshape#decoder/dense_25/Relu:activations:0)decoder/reshape_10/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
decoder/reshape_10/Reshape{
decoder/reshape_14/ShapeShapedecoder/re_lu_1/sub:z:0*
T0*
_output_shapes
:2
decoder/reshape_14/Shapeџ
&decoder/reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_14/strided_slice/stackъ
(decoder/reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_14/strided_slice/stack_1ъ
(decoder/reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_14/strided_slice/stack_2н
 decoder/reshape_14/strided_sliceStridedSlice!decoder/reshape_14/Shape:output:0/decoder/reshape_14/strided_slice/stack:output:01decoder/reshape_14/strided_slice/stack_1:output:01decoder/reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_14/strided_sliceі
"decoder/reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_14/Reshape/shape/1і
"decoder/reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_14/Reshape/shape/2і
"decoder/reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_14/Reshape/shape/3і
"decoder/reshape_14/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_14/Reshape/shape/4┘
 decoder/reshape_14/Reshape/shapePack)decoder/reshape_14/strided_slice:output:0+decoder/reshape_14/Reshape/shape/1:output:0+decoder/reshape_14/Reshape/shape/2:output:0+decoder/reshape_14/Reshape/shape/3:output:0+decoder/reshape_14/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_14/Reshape/shape┼
decoder/reshape_14/ReshapeReshapedecoder/re_lu_1/sub:z:0)decoder/reshape_14/Reshape/shape:output:0*
T0*3
_output_shapes!
:         2
decoder/reshape_14/Reshapey
decoder/reshape_13/ShapeShapedecoder/re_lu/sub:z:0*
T0*
_output_shapes
:2
decoder/reshape_13/Shapeџ
&decoder/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_13/strided_slice/stackъ
(decoder/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_13/strided_slice/stack_1ъ
(decoder/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_13/strided_slice/stack_2н
 decoder/reshape_13/strided_sliceStridedSlice!decoder/reshape_13/Shape:output:0/decoder/reshape_13/strided_slice/stack:output:01decoder/reshape_13/strided_slice/stack_1:output:01decoder/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_13/strided_sliceі
"decoder/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_13/Reshape/shape/1і
"decoder/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_13/Reshape/shape/2і
"decoder/reshape_13/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_13/Reshape/shape/3і
"decoder/reshape_13/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_13/Reshape/shape/4┘
 decoder/reshape_13/Reshape/shapePack)decoder/reshape_13/strided_slice:output:0+decoder/reshape_13/Reshape/shape/1:output:0+decoder/reshape_13/Reshape/shape/2:output:0+decoder/reshape_13/Reshape/shape/3:output:0+decoder/reshape_13/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_13/Reshape/shape├
decoder/reshape_13/ReshapeReshapedecoder/re_lu/sub:z:0)decoder/reshape_13/Reshape/shape:output:0*
T0*3
_output_shapes!
:         2
decoder/reshape_13/Reshapeћ
decoder/softmax/SoftmaxSoftmax#decoder/reshape_10/Reshape:output:0*
T0*'
_output_shapes
:         2
decoder/softmax/Softmaxі
IdentityIdentity#decoder/reshape_13/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identityј

Identity_1Identity#decoder/reshape_14/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:         2

Identity_1ђ

Identity_2Identity!decoder/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         2

Identity_2Њ
NoOpNoOp(^decoder/dense_23/BiasAdd/ReadVariableOp'^decoder/dense_23/MatMul/ReadVariableOp(^decoder/dense_24/BiasAdd/ReadVariableOp'^decoder/dense_24/MatMul/ReadVariableOp(^decoder/dense_25/BiasAdd/ReadVariableOp'^decoder/dense_25/MatMul/ReadVariableOp(^decoder/dense_30/BiasAdd/ReadVariableOp'^decoder/dense_30/MatMul/ReadVariableOp(^decoder/dense_31/BiasAdd/ReadVariableOp'^decoder/dense_31/MatMul/ReadVariableOp(^decoder/dense_32/BiasAdd/ReadVariableOp'^decoder/dense_32/MatMul/ReadVariableOp(^decoder/dense_33/BiasAdd/ReadVariableOp'^decoder/dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 2R
'decoder/dense_23/BiasAdd/ReadVariableOp'decoder/dense_23/BiasAdd/ReadVariableOp2P
&decoder/dense_23/MatMul/ReadVariableOp&decoder/dense_23/MatMul/ReadVariableOp2R
'decoder/dense_24/BiasAdd/ReadVariableOp'decoder/dense_24/BiasAdd/ReadVariableOp2P
&decoder/dense_24/MatMul/ReadVariableOp&decoder/dense_24/MatMul/ReadVariableOp2R
'decoder/dense_25/BiasAdd/ReadVariableOp'decoder/dense_25/BiasAdd/ReadVariableOp2P
&decoder/dense_25/MatMul/ReadVariableOp&decoder/dense_25/MatMul/ReadVariableOp2R
'decoder/dense_30/BiasAdd/ReadVariableOp'decoder/dense_30/BiasAdd/ReadVariableOp2P
&decoder/dense_30/MatMul/ReadVariableOp&decoder/dense_30/MatMul/ReadVariableOp2R
'decoder/dense_31/BiasAdd/ReadVariableOp'decoder/dense_31/BiasAdd/ReadVariableOp2P
&decoder/dense_31/MatMul/ReadVariableOp&decoder/dense_31/MatMul/ReadVariableOp2R
'decoder/dense_32/BiasAdd/ReadVariableOp'decoder/dense_32/BiasAdd/ReadVariableOp2P
&decoder/dense_32/MatMul/ReadVariableOp&decoder/dense_32/MatMul/ReadVariableOp2R
'decoder/dense_33/BiasAdd/ReadVariableOp'decoder/dense_33/BiasAdd/ReadVariableOp2P
&decoder/dense_33/MatMul/ReadVariableOp&decoder/dense_33/MatMul/ReadVariableOp:P L
'
_output_shapes
:          
!
_user_specified_name	input_6
й
Ѓ
*__inference_decoder_layer_call_fn_10602774
input_6
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:	И
	unknown_8:	И
	unknown_9:	И

unknown_10:	И

unknown_11:

unknown_12:
identity

identity_1

identity_2ѕбStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_106027392
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityІ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*3
_output_shapes!
:         2

Identity_1І

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*3
_output_shapes!
:         2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:          
!
_user_specified_name	input_6
║
ѓ
*__inference_decoder_layer_call_fn_10603416

inputs
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:	И
	unknown_8:	И
	unknown_9:	И

unknown_10:	И

unknown_11:

unknown_12:
identity

identity_1

identity_2ѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:         :         :         *0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_106027392
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityІ

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*3
_output_shapes!
:         2

Identity_1І

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*3
_output_shapes!
:         2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:          : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ю
d
H__inference_reshape_13_layer_call_and_return_conditional_losses_10602727

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3d
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/4н
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:         2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
┼
F
*__inference_re_lu_1_layer_call_fn_10603646

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_re_lu_1_layer_call_and_return_conditional_losses_106026632
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         И2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
К
I
-__inference_reshape_10_layer_call_fn_10603608

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_106026932
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ю
d
H__inference_reshape_14_layer_call_and_return_conditional_losses_10602710

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3d
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/4н
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:         2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
т
a
E__inference_softmax_layer_call_and_return_conditional_losses_10602734

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┴
D
(__inference_re_lu_layer_call_fn_10603627

inputs
identity┼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         И* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_re_lu_layer_call_and_return_conditional_losses_106026792
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         И2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs
р
I
-__inference_reshape_13_layer_call_fn_10603676

inputs
identityН
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_reshape_13_layer_call_and_return_conditional_losses_106027272
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         И:P L
(
_output_shapes
:         И
 
_user_specified_nameinputs"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┬
serving_default«
;
input_60
serving_default_input_6:0          J

reshape_13<
StatefulPartitionedCall:0         J

reshape_14<
StatefulPartitionedCall:1         ;
softmax0
StatefulPartitionedCall:2         tensorflow/serving/predict:╩╦
К
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Ю_default_save_signature
+ъ&call_and_return_all_conditional_losses
Ъ__call__"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
й

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+а&call_and_return_all_conditional_losses
А__call__"
_tf_keras_layer
й

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
+б&call_and_return_all_conditional_losses
Б__call__"
_tf_keras_layer
й

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
+ц&call_and_return_all_conditional_losses
Ц__call__"
_tf_keras_layer
й

'kernel
(bias
)trainable_variables
*regularization_losses
+	variables
,	keras_api
+д&call_and_return_all_conditional_losses
Д__call__"
_tf_keras_layer
й

-kernel
.bias
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+е&call_and_return_all_conditional_losses
Е__call__"
_tf_keras_layer
й

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
+ф&call_and_return_all_conditional_losses
Ф__call__"
_tf_keras_layer
й

9kernel
:bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+г&call_and_return_all_conditional_losses
Г__call__"
_tf_keras_layer
Д
?trainable_variables
@regularization_losses
A	variables
B	keras_api
+«&call_and_return_all_conditional_losses
»__call__"
_tf_keras_layer
Д
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"
_tf_keras_layer
Д
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"
_tf_keras_layer
Д
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+┤&call_and_return_all_conditional_losses
х__call__"
_tf_keras_layer
Д
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
+Х&call_and_return_all_conditional_losses
и__call__"
_tf_keras_layer
Д
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+И&call_and_return_all_conditional_losses
╣__call__"
_tf_keras_layer
є
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13"
trackable_list_wrapper
 "
trackable_list_wrapper
є
0
1
2
3
!4
"5
'6
(7
-8
.9
310
411
912
:13"
trackable_list_wrapper
╬
trainable_variables
regularization_losses

Wlayers
	variables
Xlayer_regularization_losses
Ymetrics
Zlayer_metrics
[non_trainable_variables
Ъ__call__
Ю_default_save_signature
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
-
║serving_default"
signature_map
 "
trackable_list_wrapper
!:  2dense_23/kernel
: 2dense_23/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
trainable_variables
regularization_losses

\layers
]layer_regularization_losses
	variables
^metrics
_layer_metrics
`non_trainable_variables
А__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_24/kernel
:2dense_24/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
trainable_variables
regularization_losses

alayers
blayer_regularization_losses
	variables
cmetrics
dlayer_metrics
enon_trainable_variables
Б__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_30/kernel
:2dense_30/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
░
#trainable_variables
$regularization_losses

flayers
glayer_regularization_losses
%	variables
hmetrics
ilayer_metrics
jnon_trainable_variables
Ц__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_32/kernel
:2dense_32/bias
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
░
)trainable_variables
*regularization_losses

klayers
llayer_regularization_losses
+	variables
mmetrics
nlayer_metrics
onon_trainable_variables
Д__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
!:2dense_25/kernel
:2dense_25/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
░
/trainable_variables
0regularization_losses

players
qlayer_regularization_losses
1	variables
rmetrics
slayer_metrics
tnon_trainable_variables
Е__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
": 	И2dense_31/kernel
:И2dense_31/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
░
5trainable_variables
6regularization_losses

ulayers
vlayer_regularization_losses
7	variables
wmetrics
xlayer_metrics
ynon_trainable_variables
Ф__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
": 	И2dense_33/kernel
:И2dense_33/bias
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
░
;trainable_variables
<regularization_losses

zlayers
{layer_regularization_losses
=	variables
|metrics
}layer_metrics
~non_trainable_variables
Г__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┤
?trainable_variables
@regularization_losses

layers
 ђlayer_regularization_losses
A	variables
Ђmetrics
ѓlayer_metrics
Ѓnon_trainable_variables
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ctrainable_variables
Dregularization_losses
ёlayers
 Ёlayer_regularization_losses
E	variables
єmetrics
Єlayer_metrics
ѕnon_trainable_variables
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Gtrainable_variables
Hregularization_losses
Ѕlayers
 іlayer_regularization_losses
I	variables
Іmetrics
їlayer_metrics
Їnon_trainable_variables
│__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ktrainable_variables
Lregularization_losses
јlayers
 Јlayer_regularization_losses
M	variables
љmetrics
Љlayer_metrics
њnon_trainable_variables
х__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Otrainable_variables
Pregularization_losses
Њlayers
 ћlayer_regularization_losses
Q	variables
Ћmetrics
ќlayer_metrics
Ќnon_trainable_variables
и__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Strainable_variables
Tregularization_losses
ўlayers
 Ўlayer_regularization_losses
U	variables
џmetrics
Џlayer_metrics
юnon_trainable_variables
╣__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
є
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
╬B╦
#__inference__wrapped_model_10602525input_6"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Р2▀
E__inference_decoder_layer_call_and_return_conditional_losses_10603275
E__inference_decoder_layer_call_and_return_conditional_losses_10603379
E__inference_decoder_layer_call_and_return_conditional_losses_10603085
E__inference_decoder_layer_call_and_return_conditional_losses_10603132└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ш2з
*__inference_decoder_layer_call_fn_10602774
*__inference_decoder_layer_call_fn_10603416
*__inference_decoder_layer_call_fn_10603453
*__inference_decoder_layer_call_fn_10603038└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_dense_23_layer_call_and_return_conditional_losses_10603464б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_23_layer_call_fn_10603473б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_24_layer_call_and_return_conditional_losses_10603484б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_24_layer_call_fn_10603493б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_30_layer_call_and_return_conditional_losses_10603504б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_30_layer_call_fn_10603513б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_32_layer_call_and_return_conditional_losses_10603524б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_32_layer_call_fn_10603533б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_25_layer_call_and_return_conditional_losses_10603544б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_25_layer_call_fn_10603553б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_31_layer_call_and_return_conditional_losses_10603563б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_31_layer_call_fn_10603572б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_dense_33_layer_call_and_return_conditional_losses_10603582б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_dense_33_layer_call_fn_10603591б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_reshape_10_layer_call_and_return_conditional_losses_10603603б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_reshape_10_layer_call_fn_10603608б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_re_lu_layer_call_and_return_conditional_losses_10603622б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_re_lu_layer_call_fn_10603627б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_re_lu_1_layer_call_and_return_conditional_losses_10603641б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_re_lu_1_layer_call_fn_10603646б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ч2щ
E__inference_softmax_layer_call_and_return_conditional_losses_10603651»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
р2я
*__inference_softmax_layer_call_fn_10603656»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_reshape_13_layer_call_and_return_conditional_losses_10603671б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_reshape_13_layer_call_fn_10603676б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_reshape_14_layer_call_and_return_conditional_losses_10603691б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_reshape_14_layer_call_fn_10603696б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
═B╩
&__inference_signature_wrapper_10603171input_6"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Ъ
#__inference__wrapped_model_10602525э'(!"9:34-.0б-
&б#
!і
input_6          
ф "▓ф«
>

reshape_130і-

reshape_13         
>

reshape_140і-

reshape_14         
,
softmax!і
softmax         Ў
E__inference_decoder_layer_call_and_return_conditional_losses_10603085¤'(!"9:34-.8б5
.б+
!і
input_6          
p 

 
ф "ѓб
xџu
і
0/0         
)і&
0/1         
)і&
0/2         
џ Ў
E__inference_decoder_layer_call_and_return_conditional_losses_10603132¤'(!"9:34-.8б5
.б+
!і
input_6          
p

 
ф "ѓб
xџu
і
0/0         
)і&
0/1         
)і&
0/2         
џ ў
E__inference_decoder_layer_call_and_return_conditional_losses_10603275╬'(!"9:34-.7б4
-б*
 і
inputs          
p 

 
ф "ѓб
xџu
і
0/0         
)і&
0/1         
)і&
0/2         
џ ў
E__inference_decoder_layer_call_and_return_conditional_losses_10603379╬'(!"9:34-.7б4
-б*
 і
inputs          
p

 
ф "ѓб
xџu
і
0/0         
)і&
0/1         
)і&
0/2         
џ ь
*__inference_decoder_layer_call_fn_10602774Й'(!"9:34-.8б5
.б+
!і
input_6          
p 

 
ф "rџo
і
0         
'і$
1         
'і$
2         ь
*__inference_decoder_layer_call_fn_10603038Й'(!"9:34-.8б5
.б+
!і
input_6          
p

 
ф "rџo
і
0         
'і$
1         
'і$
2         В
*__inference_decoder_layer_call_fn_10603416й'(!"9:34-.7б4
-б*
 і
inputs          
p 

 
ф "rџo
і
0         
'і$
1         
'і$
2         В
*__inference_decoder_layer_call_fn_10603453й'(!"9:34-.7б4
-б*
 і
inputs          
p

 
ф "rџo
і
0         
'і$
1         
'і$
2         д
F__inference_dense_23_layer_call_and_return_conditional_losses_10603464\/б,
%б"
 і
inputs          
ф "%б"
і
0          
џ ~
+__inference_dense_23_layer_call_fn_10603473O/б,
%б"
 і
inputs          
ф "і          д
F__inference_dense_24_layer_call_and_return_conditional_losses_10603484\/б,
%б"
 і
inputs          
ф "%б"
і
0         
џ ~
+__inference_dense_24_layer_call_fn_10603493O/б,
%б"
 і
inputs          
ф "і         д
F__inference_dense_25_layer_call_and_return_conditional_losses_10603544\-./б,
%б"
 і
inputs         
ф "%б"
і
0         
џ ~
+__inference_dense_25_layer_call_fn_10603553O-./б,
%б"
 і
inputs         
ф "і         д
F__inference_dense_30_layer_call_and_return_conditional_losses_10603504\!"/б,
%б"
 і
inputs          
ф "%б"
і
0         
џ ~
+__inference_dense_30_layer_call_fn_10603513O!"/б,
%б"
 і
inputs          
ф "і         Д
F__inference_dense_31_layer_call_and_return_conditional_losses_10603563]34/б,
%б"
 і
inputs         
ф "&б#
і
0         И
џ 
+__inference_dense_31_layer_call_fn_10603572P34/б,
%б"
 і
inputs         
ф "і         Ид
F__inference_dense_32_layer_call_and_return_conditional_losses_10603524\'(/б,
%б"
 і
inputs          
ф "%б"
і
0         
џ ~
+__inference_dense_32_layer_call_fn_10603533O'(/б,
%б"
 і
inputs          
ф "і         Д
F__inference_dense_33_layer_call_and_return_conditional_losses_10603582]9:/б,
%б"
 і
inputs         
ф "&б#
і
0         И
џ 
+__inference_dense_33_layer_call_fn_10603591P9:/б,
%б"
 і
inputs         
ф "і         ИБ
E__inference_re_lu_1_layer_call_and_return_conditional_losses_10603641Z0б-
&б#
!і
inputs         И
ф "&б#
і
0         И
џ {
*__inference_re_lu_1_layer_call_fn_10603646M0б-
&б#
!і
inputs         И
ф "і         ИА
C__inference_re_lu_layer_call_and_return_conditional_losses_10603622Z0б-
&б#
!і
inputs         И
ф "&б#
і
0         И
џ y
(__inference_re_lu_layer_call_fn_10603627M0б-
&б#
!і
inputs         И
ф "і         Иц
H__inference_reshape_10_layer_call_and_return_conditional_losses_10603603X/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ |
-__inference_reshape_10_layer_call_fn_10603608K/б,
%б"
 і
inputs         
ф "і         ▒
H__inference_reshape_13_layer_call_and_return_conditional_losses_10603671e0б-
&б#
!і
inputs         И
ф "1б.
'і$
0         
џ Ѕ
-__inference_reshape_13_layer_call_fn_10603676X0б-
&б#
!і
inputs         И
ф "$і!         ▒
H__inference_reshape_14_layer_call_and_return_conditional_losses_10603691e0б-
&б#
!і
inputs         И
ф "1б.
'і$
0         
џ Ѕ
-__inference_reshape_14_layer_call_fn_10603696X0б-
&б#
!і
inputs         И
ф "$і!         Г
&__inference_signature_wrapper_10603171ѓ'(!"9:34-.;б8
б 
1ф.
,
input_6!і
input_6          "▓ф«
>

reshape_130і-

reshape_13         
>

reshape_140і-

reshape_14         
,
softmax!і
softmax         Ц
E__inference_softmax_layer_call_and_return_conditional_losses_10603651\3б0
)б&
 і
inputs         

 
ф "%б"
і
0         
џ }
*__inference_softmax_layer_call_fn_10603656O3б0
)б&
 і
inputs         

 
ф "і         