ˑ
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
delete_old_dirsbool(�
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
2	�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8��
z
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_23/kernel
s
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes

:@@*
dtype0
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:@*
dtype0
z
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

:@*
dtype0
r
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_26/bias
k
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes
:*
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

:@*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
:*
dtype0
z
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_30/kernel
s
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes

:@*
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
:@* 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

:@*
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
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_27/kernel
s
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes

:d*
dtype0
r
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:d*
dtype0
z
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0* 
shared_namedense_29/kernel
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes

:0*
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:0*
dtype0
{
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�N* 
shared_namedense_31/kernel
t
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes
:	�N*
dtype0
s
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�N*
shared_namedense_31/bias
l
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes	
:�N*
dtype0
{
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�N* 
shared_namedense_33/kernel
t
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes
:	�N*
dtype0
s
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�N*
shared_namedense_33/bias
l
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes	
:�N*
dtype0

NoOpNoOp
�?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�?
value�?B�> B�>
�
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
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
regularization_losses
	variables
trainable_variables
	keras_api

signatures

_init_input_shape
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
h

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
R
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
R
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
R
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
R
[regularization_losses
\trainable_variables
]	variables
^	keras_api
R
_regularization_losses
`trainable_variables
a	variables
b	keras_api
R
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
R
gregularization_losses
htrainable_variables
i	variables
j	keras_api
R
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
 
�
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
I16
J17
�
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
I16
J17
�
olayer_metrics
player_regularization_losses

qlayers
regularization_losses
rnon_trainable_variables
smetrics
	variables
trainable_variables
 
 
[Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_23/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
tlayer_metrics

ulayers
vnon_trainable_variables
regularization_losses
wmetrics
trainable_variables
xlayer_regularization_losses
	variables
[Y
VARIABLE_VALUEdense_26/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_26/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
�
ylayer_metrics

zlayers
{non_trainable_variables
!regularization_losses
|metrics
"trainable_variables
}layer_regularization_losses
#	variables
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
�
~layer_metrics

layers
�non_trainable_variables
'regularization_losses
�metrics
(trainable_variables
 �layer_regularization_losses
)	variables
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
�
�layer_metrics
�layers
�non_trainable_variables
-regularization_losses
�metrics
.trainable_variables
 �layer_regularization_losses
/	variables
[Y
VARIABLE_VALUEdense_32/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
�
�layer_metrics
�layers
�non_trainable_variables
3regularization_losses
�metrics
4trainable_variables
 �layer_regularization_losses
5	variables
[Y
VARIABLE_VALUEdense_27/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_27/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
�
�layer_metrics
�layers
�non_trainable_variables
9regularization_losses
�metrics
:trainable_variables
 �layer_regularization_losses
;	variables
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
�
�layer_metrics
�layers
�non_trainable_variables
?regularization_losses
�metrics
@trainable_variables
 �layer_regularization_losses
A	variables
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
�
�layer_metrics
�layers
�non_trainable_variables
Eregularization_losses
�metrics
Ftrainable_variables
 �layer_regularization_losses
G	variables
[Y
VARIABLE_VALUEdense_33/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

I0
J1

I0
J1
�
�layer_metrics
�layers
�non_trainable_variables
Kregularization_losses
�metrics
Ltrainable_variables
 �layer_regularization_losses
M	variables
 
 
 
�
�layer_metrics
�layers
�non_trainable_variables
Oregularization_losses
�metrics
Ptrainable_variables
 �layer_regularization_losses
Q	variables
 
 
 
�
�layer_metrics
�layers
�non_trainable_variables
Sregularization_losses
�metrics
Ttrainable_variables
 �layer_regularization_losses
U	variables
 
 
 
�
�layer_metrics
�layers
�non_trainable_variables
Wregularization_losses
�metrics
Xtrainable_variables
 �layer_regularization_losses
Y	variables
 
 
 
�
�layer_metrics
�layers
�non_trainable_variables
[regularization_losses
�metrics
\trainable_variables
 �layer_regularization_losses
]	variables
 
 
 
�
�layer_metrics
�layers
�non_trainable_variables
_regularization_losses
�metrics
`trainable_variables
 �layer_regularization_losses
a	variables
 
 
 
�
�layer_metrics
�layers
�non_trainable_variables
cregularization_losses
�metrics
dtrainable_variables
 �layer_regularization_losses
e	variables
 
 
 
�
�layer_metrics
�layers
�non_trainable_variables
gregularization_losses
�metrics
htrainable_variables
 �layer_regularization_losses
i	variables
 
 
 
�
�layer_metrics
�layers
�non_trainable_variables
kregularization_losses
�metrics
ltrainable_variables
 �layer_regularization_losses
m	variables
 
 
�
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
14
15
16
17
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
:���������@*
dtype0*
shape:���������@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_6dense_23/kerneldense_23/biasdense_32/kerneldense_32/biasdense_30/kerneldense_30/biasdense_28/kerneldense_28/biasdense_26/kerneldense_26/biasdense_33/kerneldense_33/biasdense_31/kerneldense_31/biasdense_29/kerneldense_29/biasdense_27/kerneldense_27/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *�
_output_shapesv
t:���������d:���������d:���������
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_6736827
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_6737579
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_23/kerneldense_23/biasdense_26/kerneldense_26/biasdense_28/kerneldense_28/biasdense_30/kerneldense_30/biasdense_32/kerneldense_32/biasdense_27/kerneldense_27/biasdense_29/kerneldense_29/biasdense_31/kerneldense_31/biasdense_33/kerneldense_33/bias*
Tin
2*
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_6737643��
�
�
*__inference_dense_29_layer_call_fn_6737325

inputs
unknown:0
	unknown_0:0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_67361352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_softmax_1_layer_call_and_return_conditional_losses_6737444

inputs
identity_
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:���������
2	
Softmaxm
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
E__inference_dense_31_layer_call_and_return_conditional_losses_6736118

inputs1
matmul_readvariableop_resource:	�N.
biasadd_readvariableop_resource:	�N
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������N2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_softmax_1_layer_call_and_return_conditional_losses_6736268

inputs
identity_
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:���������
2	
Softmaxm
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
E__inference_dense_28_layer_call_and_return_conditional_losses_6736069

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
D__inference_decoder_layer_call_and_return_conditional_losses_6737091

inputs9
'dense_23_matmul_readvariableop_resource:@@6
(dense_23_biasadd_readvariableop_resource:@9
'dense_32_matmul_readvariableop_resource:@6
(dense_32_biasadd_readvariableop_resource:9
'dense_30_matmul_readvariableop_resource:@6
(dense_30_biasadd_readvariableop_resource:9
'dense_28_matmul_readvariableop_resource:@6
(dense_28_biasadd_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource:@6
(dense_26_biasadd_readvariableop_resource::
'dense_33_matmul_readvariableop_resource:	�N7
(dense_33_biasadd_readvariableop_resource:	�N:
'dense_31_matmul_readvariableop_resource:	�N7
(dense_31_biasadd_readvariableop_resource:	�N9
'dense_29_matmul_readvariableop_resource:06
(dense_29_biasadd_readvariableop_resource:09
'dense_27_matmul_readvariableop_resource:d6
(dense_27_biasadd_readvariableop_resource:d
identity

identity_1

identity_2

identity_3��dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�dense_30/BiasAdd/ReadVariableOp�dense_30/MatMul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOp�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_23/MatMul/ReadVariableOp�
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_23/MatMul�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOp�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_23/BiasAdds
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_23/Relu�
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_32/MatMul/ReadVariableOp�
dense_32/MatMulMatMuldense_23/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_32/MatMul�
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOp�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_32/BiasAdds
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_32/Relu�
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_30/MatMul/ReadVariableOp�
dense_30/MatMulMatMuldense_23/Relu:activations:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_30/MatMul�
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_30/BiasAdd/ReadVariableOp�
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_30/Relu�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_28/MatMul/ReadVariableOp�
dense_28/MatMulMatMuldense_23/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_28/MatMul�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_28/BiasAdd/ReadVariableOp�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_28/Relu�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMuldense_23/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_26/Relu�
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02 
dense_33/MatMul/ReadVariableOp�
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
dense_33/MatMul�
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02!
dense_33/BiasAdd/ReadVariableOp�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
dense_33/BiasAdd�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02 
dense_31/MatMul/ReadVariableOp�
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
dense_31/MatMul�
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02!
dense_31/BiasAdd/ReadVariableOp�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
dense_31/BiasAdd�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02
dense_29/BiasAdds
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������02
dense_29/Relu�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_27/Reluo
re_lu_1/NegNegdense_33/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
re_lu_1/Negh
re_lu_1/ReluRelure_lu_1/Neg:y:0*
T0*(
_output_shapes
:����������N2
re_lu_1/Reluv
re_lu_1/Relu_1Reludense_33/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
re_lu_1/Relu_1c
re_lu_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
re_lu_1/Constg
re_lu_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
re_lu_1/Const_1�
re_lu_1/clip_by_value/MinimumMinimumre_lu_1/Relu_1:activations:0re_lu_1/Const:output:0*
T0*(
_output_shapes
:����������N2
re_lu_1/clip_by_value/Minimum�
re_lu_1/clip_by_valueMaximum!re_lu_1/clip_by_value/Minimum:z:0re_lu_1/Const_1:output:0*
T0*(
_output_shapes
:����������N2
re_lu_1/clip_by_valueg
re_lu_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2
re_lu_1/Const_2�
re_lu_1/mulMulre_lu_1/Const_2:output:0re_lu_1/Relu:activations:0*
T0*(
_output_shapes
:����������N2
re_lu_1/mul�
re_lu_1/subSubre_lu_1/clip_by_value:z:0re_lu_1/mul:z:0*
T0*(
_output_shapes
:����������N2
re_lu_1/subk
	re_lu/NegNegdense_31/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
	re_lu/Negb

re_lu/ReluRelure_lu/Neg:y:0*
T0*(
_output_shapes
:����������N2

re_lu/Relur
re_lu/Relu_1Reludense_31/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
re_lu/Relu_1_
re_lu/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
re_lu/Constc
re_lu/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
re_lu/Const_1�
re_lu/clip_by_value/MinimumMinimumre_lu/Relu_1:activations:0re_lu/Const:output:0*
T0*(
_output_shapes
:����������N2
re_lu/clip_by_value/Minimum�
re_lu/clip_by_valueMaximumre_lu/clip_by_value/Minimum:z:0re_lu/Const_1:output:0*
T0*(
_output_shapes
:����������N2
re_lu/clip_by_valuec
re_lu/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2
re_lu/Const_2�
	re_lu/mulMulre_lu/Const_2:output:0re_lu/Relu:activations:0*
T0*(
_output_shapes
:����������N2
	re_lu/mulx
	re_lu/subSubre_lu/clip_by_value:z:0re_lu/mul:z:0*
T0*(
_output_shapes
:����������N2
	re_lu/subo
reshape_12/ShapeShapedense_29/Relu:activations:0*
T0*
_output_shapes
:2
reshape_12/Shape�
reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_12/strided_slice/stack�
 reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_1�
 reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_2�
reshape_12/strided_sliceStridedSlicereshape_12/Shape:output:0'reshape_12/strided_slice/stack:output:0)reshape_12/strided_slice/stack_1:output:0)reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_12/strided_slicez
reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/1z
reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/2z
reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/3�
reshape_12/Reshape/shapePack!reshape_12/strided_slice:output:0#reshape_12/Reshape/shape/1:output:0#reshape_12/Reshape/shape/2:output:0#reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_12/Reshape/shape�
reshape_12/ReshapeReshapedense_29/Relu:activations:0!reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape_12/Reshapeo
reshape_11/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
reshape_11/Shape�
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_11/strided_slice/stack�
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_1�
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_2�
reshape_11/strided_sliceStridedSlicereshape_11/Shape:output:0'reshape_11/strided_slice/stack:output:0)reshape_11/strided_slice/stack_1:output:0)reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_11/strided_slicez
reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/1z
reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/2z
reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_11/Reshape/shape/3�
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0#reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_11/Reshape/shape�
reshape_11/ReshapeReshapedense_27/Relu:activations:0!reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:���������
2
reshape_11/Reshapec
reshape_14/ShapeShapere_lu_1/sub:z:0*
T0*
_output_shapes
:2
reshape_14/Shape�
reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_14/strided_slice/stack�
 reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_1�
 reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_2�
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
value	B :d2
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
reshape_14/Reshape/shape/4�
reshape_14/Reshape/shapePack!reshape_14/strided_slice:output:0#reshape_14/Reshape/shape/1:output:0#reshape_14/Reshape/shape/2:output:0#reshape_14/Reshape/shape/3:output:0#reshape_14/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape_14/Reshape/shape�
reshape_14/ReshapeReshapere_lu_1/sub:z:0!reshape_14/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������d2
reshape_14/Reshapea
reshape_13/ShapeShapere_lu/sub:z:0*
T0*
_output_shapes
:2
reshape_13/Shape�
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack�
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1�
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2�
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
value	B :d2
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
reshape_13/Reshape/shape/4�
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0#reshape_13/Reshape/shape/3:output:0#reshape_13/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape�
reshape_13/ReshapeReshapere_lu/sub:z:0!reshape_13/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������d2
reshape_13/Reshape�
softmax_2/SoftmaxSoftmaxreshape_12/Reshape:output:0*
T0*/
_output_shapes
:���������2
softmax_2/Softmax�
softmax_1/SoftmaxSoftmaxreshape_11/Reshape:output:0*
T0*/
_output_shapes
:���������
2
softmax_1/Softmax~
IdentityIdentitysoftmax_1/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identitysoftmax_2/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identityreshape_13/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identityreshape_14/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3�
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
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
:���������@
 
_user_specified_nameinputs
�.
�
 __inference__traced_save_6737579
file_prefix.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :@@:@:@::@::@::@::d:d:0:0:	�N:�N:	�N:�N: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$	 

_output_shapes

:@: 


_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:0: 

_output_shapes
:0:%!

_output_shapes
:	�N:!

_output_shapes	
:�N:%!

_output_shapes
:	�N:!

_output_shapes	
:�N:

_output_shapes
: 
�
H
,__inference_reshape_13_layer_call_fn_6737479

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_67362542
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�
�
E__inference_dense_30_layer_call_and_return_conditional_losses_6737256

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�N
�
D__inference_decoder_layer_call_and_return_conditional_losses_6736566

inputs"
dense_23_6736509:@@
dense_23_6736511:@"
dense_32_6736514:@
dense_32_6736516:"
dense_30_6736519:@
dense_30_6736521:"
dense_28_6736524:@
dense_28_6736526:"
dense_26_6736529:@
dense_26_6736531:#
dense_33_6736534:	�N
dense_33_6736536:	�N#
dense_31_6736539:	�N
dense_31_6736541:	�N"
dense_29_6736544:0
dense_29_6736546:0"
dense_27_6736549:d
dense_27_6736551:d
identity

identity_1

identity_2

identity_3�� dense_23/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall� dense_30/StatefulPartitionedCall� dense_31/StatefulPartitionedCall� dense_32/StatefulPartitionedCall� dense_33/StatefulPartitionedCall�
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_6736509dense_23_6736511*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_67360182"
 dense_23/StatefulPartitionedCall�
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_32_6736514dense_32_6736516*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_67360352"
 dense_32/StatefulPartitionedCall�
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_30_6736519dense_30_6736521*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_67360522"
 dense_30/StatefulPartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_28_6736524dense_28_6736526*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_67360692"
 dense_28/StatefulPartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_26_6736529dense_26_6736531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_67360862"
 dense_26/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_6736534dense_33_6736536*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_67361022"
 dense_33/StatefulPartitionedCall�
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_6736539dense_31_6736541*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_67361182"
 dense_31/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_6736544dense_29_6736546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_67361352"
 dense_29/StatefulPartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_6736549dense_27_6736551*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_67361522"
 dense_27/StatefulPartitionedCall�
re_lu_1/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_67361722
re_lu_1/PartitionedCall�
re_lu/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_67361882
re_lu/PartitionedCall�
reshape_12/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_12_layer_call_and_return_conditional_losses_67362042
reshape_12/PartitionedCall�
reshape_11/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_11_layer_call_and_return_conditional_losses_67362202
reshape_11/PartitionedCall�
reshape_14/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_14_layer_call_and_return_conditional_losses_67362372
reshape_14/PartitionedCall�
reshape_13/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_67362542
reshape_13/PartitionedCall�
softmax_2/PartitionedCallPartitionedCall#reshape_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_2_layer_call_and_return_conditional_losses_67362612
softmax_2/PartitionedCall�
softmax_1/PartitionedCallPartitionedCall#reshape_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_1_layer_call_and_return_conditional_losses_67362682
softmax_1/PartitionedCall�
IdentityIdentity"softmax_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity"softmax_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identity#reshape_13/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identity#reshape_14/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3�
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
"__inference__wrapped_model_6736000
input_6A
/decoder_dense_23_matmul_readvariableop_resource:@@>
0decoder_dense_23_biasadd_readvariableop_resource:@A
/decoder_dense_32_matmul_readvariableop_resource:@>
0decoder_dense_32_biasadd_readvariableop_resource:A
/decoder_dense_30_matmul_readvariableop_resource:@>
0decoder_dense_30_biasadd_readvariableop_resource:A
/decoder_dense_28_matmul_readvariableop_resource:@>
0decoder_dense_28_biasadd_readvariableop_resource:A
/decoder_dense_26_matmul_readvariableop_resource:@>
0decoder_dense_26_biasadd_readvariableop_resource:B
/decoder_dense_33_matmul_readvariableop_resource:	�N?
0decoder_dense_33_biasadd_readvariableop_resource:	�NB
/decoder_dense_31_matmul_readvariableop_resource:	�N?
0decoder_dense_31_biasadd_readvariableop_resource:	�NA
/decoder_dense_29_matmul_readvariableop_resource:0>
0decoder_dense_29_biasadd_readvariableop_resource:0A
/decoder_dense_27_matmul_readvariableop_resource:d>
0decoder_dense_27_biasadd_readvariableop_resource:d
identity

identity_1

identity_2

identity_3��'decoder/dense_23/BiasAdd/ReadVariableOp�&decoder/dense_23/MatMul/ReadVariableOp�'decoder/dense_26/BiasAdd/ReadVariableOp�&decoder/dense_26/MatMul/ReadVariableOp�'decoder/dense_27/BiasAdd/ReadVariableOp�&decoder/dense_27/MatMul/ReadVariableOp�'decoder/dense_28/BiasAdd/ReadVariableOp�&decoder/dense_28/MatMul/ReadVariableOp�'decoder/dense_29/BiasAdd/ReadVariableOp�&decoder/dense_29/MatMul/ReadVariableOp�'decoder/dense_30/BiasAdd/ReadVariableOp�&decoder/dense_30/MatMul/ReadVariableOp�'decoder/dense_31/BiasAdd/ReadVariableOp�&decoder/dense_31/MatMul/ReadVariableOp�'decoder/dense_32/BiasAdd/ReadVariableOp�&decoder/dense_32/MatMul/ReadVariableOp�'decoder/dense_33/BiasAdd/ReadVariableOp�&decoder/dense_33/MatMul/ReadVariableOp�
&decoder/dense_23/MatMul/ReadVariableOpReadVariableOp/decoder_dense_23_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02(
&decoder/dense_23/MatMul/ReadVariableOp�
decoder/dense_23/MatMulMatMulinput_6.decoder/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
decoder/dense_23/MatMul�
'decoder/dense_23/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'decoder/dense_23/BiasAdd/ReadVariableOp�
decoder/dense_23/BiasAddBiasAdd!decoder/dense_23/MatMul:product:0/decoder/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
decoder/dense_23/BiasAdd�
decoder/dense_23/ReluRelu!decoder/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
decoder/dense_23/Relu�
&decoder/dense_32/MatMul/ReadVariableOpReadVariableOp/decoder_dense_32_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&decoder/dense_32/MatMul/ReadVariableOp�
decoder/dense_32/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/dense_32/MatMul�
'decoder/dense_32/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_32/BiasAdd/ReadVariableOp�
decoder/dense_32/BiasAddBiasAdd!decoder/dense_32/MatMul:product:0/decoder/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/dense_32/BiasAdd�
decoder/dense_32/ReluRelu!decoder/dense_32/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
decoder/dense_32/Relu�
&decoder/dense_30/MatMul/ReadVariableOpReadVariableOp/decoder_dense_30_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&decoder/dense_30/MatMul/ReadVariableOp�
decoder/dense_30/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/dense_30/MatMul�
'decoder/dense_30/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_30/BiasAdd/ReadVariableOp�
decoder/dense_30/BiasAddBiasAdd!decoder/dense_30/MatMul:product:0/decoder/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/dense_30/BiasAdd�
decoder/dense_30/ReluRelu!decoder/dense_30/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
decoder/dense_30/Relu�
&decoder/dense_28/MatMul/ReadVariableOpReadVariableOp/decoder_dense_28_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&decoder/dense_28/MatMul/ReadVariableOp�
decoder/dense_28/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/dense_28/MatMul�
'decoder/dense_28/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_28/BiasAdd/ReadVariableOp�
decoder/dense_28/BiasAddBiasAdd!decoder/dense_28/MatMul:product:0/decoder/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/dense_28/BiasAdd�
decoder/dense_28/ReluRelu!decoder/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
decoder/dense_28/Relu�
&decoder/dense_26/MatMul/ReadVariableOpReadVariableOp/decoder_dense_26_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02(
&decoder/dense_26/MatMul/ReadVariableOp�
decoder/dense_26/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/dense_26/MatMul�
'decoder/dense_26/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_26/BiasAdd/ReadVariableOp�
decoder/dense_26/BiasAddBiasAdd!decoder/dense_26/MatMul:product:0/decoder/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
decoder/dense_26/BiasAdd�
decoder/dense_26/ReluRelu!decoder/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
decoder/dense_26/Relu�
&decoder/dense_33/MatMul/ReadVariableOpReadVariableOp/decoder_dense_33_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02(
&decoder/dense_33/MatMul/ReadVariableOp�
decoder/dense_33/MatMulMatMul#decoder/dense_32/Relu:activations:0.decoder/dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
decoder/dense_33/MatMul�
'decoder/dense_33/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_33_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02)
'decoder/dense_33/BiasAdd/ReadVariableOp�
decoder/dense_33/BiasAddBiasAdd!decoder/dense_33/MatMul:product:0/decoder/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
decoder/dense_33/BiasAdd�
&decoder/dense_31/MatMul/ReadVariableOpReadVariableOp/decoder_dense_31_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02(
&decoder/dense_31/MatMul/ReadVariableOp�
decoder/dense_31/MatMulMatMul#decoder/dense_30/Relu:activations:0.decoder/dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
decoder/dense_31/MatMul�
'decoder/dense_31/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_31_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02)
'decoder/dense_31/BiasAdd/ReadVariableOp�
decoder/dense_31/BiasAddBiasAdd!decoder/dense_31/MatMul:product:0/decoder/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
decoder/dense_31/BiasAdd�
&decoder/dense_29/MatMul/ReadVariableOpReadVariableOp/decoder_dense_29_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02(
&decoder/dense_29/MatMul/ReadVariableOp�
decoder/dense_29/MatMulMatMul#decoder/dense_28/Relu:activations:0.decoder/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02
decoder/dense_29/MatMul�
'decoder/dense_29/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_29_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'decoder/dense_29/BiasAdd/ReadVariableOp�
decoder/dense_29/BiasAddBiasAdd!decoder/dense_29/MatMul:product:0/decoder/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02
decoder/dense_29/BiasAdd�
decoder/dense_29/ReluRelu!decoder/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������02
decoder/dense_29/Relu�
&decoder/dense_27/MatMul/ReadVariableOpReadVariableOp/decoder_dense_27_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02(
&decoder/dense_27/MatMul/ReadVariableOp�
decoder/dense_27/MatMulMatMul#decoder/dense_26/Relu:activations:0.decoder/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
decoder/dense_27/MatMul�
'decoder/dense_27/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_27_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'decoder/dense_27/BiasAdd/ReadVariableOp�
decoder/dense_27/BiasAddBiasAdd!decoder/dense_27/MatMul:product:0/decoder/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
decoder/dense_27/BiasAdd�
decoder/dense_27/ReluRelu!decoder/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
decoder/dense_27/Relu�
decoder/re_lu_1/NegNeg!decoder/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu_1/Neg�
decoder/re_lu_1/ReluReludecoder/re_lu_1/Neg:y:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu_1/Relu�
decoder/re_lu_1/Relu_1Relu!decoder/dense_33/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu_1/Relu_1s
decoder/re_lu_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
decoder/re_lu_1/Constw
decoder/re_lu_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
decoder/re_lu_1/Const_1�
%decoder/re_lu_1/clip_by_value/MinimumMinimum$decoder/re_lu_1/Relu_1:activations:0decoder/re_lu_1/Const:output:0*
T0*(
_output_shapes
:����������N2'
%decoder/re_lu_1/clip_by_value/Minimum�
decoder/re_lu_1/clip_by_valueMaximum)decoder/re_lu_1/clip_by_value/Minimum:z:0 decoder/re_lu_1/Const_1:output:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu_1/clip_by_valuew
decoder/re_lu_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2
decoder/re_lu_1/Const_2�
decoder/re_lu_1/mulMul decoder/re_lu_1/Const_2:output:0"decoder/re_lu_1/Relu:activations:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu_1/mul�
decoder/re_lu_1/subSub!decoder/re_lu_1/clip_by_value:z:0decoder/re_lu_1/mul:z:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu_1/sub�
decoder/re_lu/NegNeg!decoder/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu/Negz
decoder/re_lu/ReluReludecoder/re_lu/Neg:y:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu/Relu�
decoder/re_lu/Relu_1Relu!decoder/dense_31/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu/Relu_1o
decoder/re_lu/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
decoder/re_lu/Consts
decoder/re_lu/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
decoder/re_lu/Const_1�
#decoder/re_lu/clip_by_value/MinimumMinimum"decoder/re_lu/Relu_1:activations:0decoder/re_lu/Const:output:0*
T0*(
_output_shapes
:����������N2%
#decoder/re_lu/clip_by_value/Minimum�
decoder/re_lu/clip_by_valueMaximum'decoder/re_lu/clip_by_value/Minimum:z:0decoder/re_lu/Const_1:output:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu/clip_by_values
decoder/re_lu/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2
decoder/re_lu/Const_2�
decoder/re_lu/mulMuldecoder/re_lu/Const_2:output:0 decoder/re_lu/Relu:activations:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu/mul�
decoder/re_lu/subSubdecoder/re_lu/clip_by_value:z:0decoder/re_lu/mul:z:0*
T0*(
_output_shapes
:����������N2
decoder/re_lu/sub�
decoder/reshape_12/ShapeShape#decoder/dense_29/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_12/Shape�
&decoder/reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_12/strided_slice/stack�
(decoder/reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_12/strided_slice/stack_1�
(decoder/reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_12/strided_slice/stack_2�
 decoder/reshape_12/strided_sliceStridedSlice!decoder/reshape_12/Shape:output:0/decoder/reshape_12/strided_slice/stack:output:01decoder/reshape_12/strided_slice/stack_1:output:01decoder/reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_12/strided_slice�
"decoder/reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/1�
"decoder/reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/2�
"decoder/reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/3�
 decoder/reshape_12/Reshape/shapePack)decoder/reshape_12/strided_slice:output:0+decoder/reshape_12/Reshape/shape/1:output:0+decoder/reshape_12/Reshape/shape/2:output:0+decoder/reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_12/Reshape/shape�
decoder/reshape_12/ReshapeReshape#decoder/dense_29/Relu:activations:0)decoder/reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
decoder/reshape_12/Reshape�
decoder/reshape_11/ShapeShape#decoder/dense_27/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_11/Shape�
&decoder/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_11/strided_slice/stack�
(decoder/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_11/strided_slice/stack_1�
(decoder/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_11/strided_slice/stack_2�
 decoder/reshape_11/strided_sliceStridedSlice!decoder/reshape_11/Shape:output:0/decoder/reshape_11/strided_slice/stack:output:01decoder/reshape_11/strided_slice/stack_1:output:01decoder/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_11/strided_slice�
"decoder/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_11/Reshape/shape/1�
"decoder/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_11/Reshape/shape/2�
"decoder/reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
2$
"decoder/reshape_11/Reshape/shape/3�
 decoder/reshape_11/Reshape/shapePack)decoder/reshape_11/strided_slice:output:0+decoder/reshape_11/Reshape/shape/1:output:0+decoder/reshape_11/Reshape/shape/2:output:0+decoder/reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_11/Reshape/shape�
decoder/reshape_11/ReshapeReshape#decoder/dense_27/Relu:activations:0)decoder/reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:���������
2
decoder/reshape_11/Reshape{
decoder/reshape_14/ShapeShapedecoder/re_lu_1/sub:z:0*
T0*
_output_shapes
:2
decoder/reshape_14/Shape�
&decoder/reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_14/strided_slice/stack�
(decoder/reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_14/strided_slice/stack_1�
(decoder/reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_14/strided_slice/stack_2�
 decoder/reshape_14/strided_sliceStridedSlice!decoder/reshape_14/Shape:output:0/decoder/reshape_14/strided_slice/stack:output:01decoder/reshape_14/strided_slice/stack_1:output:01decoder/reshape_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_14/strided_slice�
"decoder/reshape_14/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_14/Reshape/shape/1�
"decoder/reshape_14/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2$
"decoder/reshape_14/Reshape/shape/2�
"decoder/reshape_14/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_14/Reshape/shape/3�
"decoder/reshape_14/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_14/Reshape/shape/4�
 decoder/reshape_14/Reshape/shapePack)decoder/reshape_14/strided_slice:output:0+decoder/reshape_14/Reshape/shape/1:output:0+decoder/reshape_14/Reshape/shape/2:output:0+decoder/reshape_14/Reshape/shape/3:output:0+decoder/reshape_14/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_14/Reshape/shape�
decoder/reshape_14/ReshapeReshapedecoder/re_lu_1/sub:z:0)decoder/reshape_14/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������d2
decoder/reshape_14/Reshapey
decoder/reshape_13/ShapeShapedecoder/re_lu/sub:z:0*
T0*
_output_shapes
:2
decoder/reshape_13/Shape�
&decoder/reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_13/strided_slice/stack�
(decoder/reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_13/strided_slice/stack_1�
(decoder/reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_13/strided_slice/stack_2�
 decoder/reshape_13/strided_sliceStridedSlice!decoder/reshape_13/Shape:output:0/decoder/reshape_13/strided_slice/stack:output:01decoder/reshape_13/strided_slice/stack_1:output:01decoder/reshape_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_13/strided_slice�
"decoder/reshape_13/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_13/Reshape/shape/1�
"decoder/reshape_13/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :d2$
"decoder/reshape_13/Reshape/shape/2�
"decoder/reshape_13/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_13/Reshape/shape/3�
"decoder/reshape_13/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_13/Reshape/shape/4�
 decoder/reshape_13/Reshape/shapePack)decoder/reshape_13/strided_slice:output:0+decoder/reshape_13/Reshape/shape/1:output:0+decoder/reshape_13/Reshape/shape/2:output:0+decoder/reshape_13/Reshape/shape/3:output:0+decoder/reshape_13/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_13/Reshape/shape�
decoder/reshape_13/ReshapeReshapedecoder/re_lu/sub:z:0)decoder/reshape_13/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������d2
decoder/reshape_13/Reshape�
decoder/softmax_2/SoftmaxSoftmax#decoder/reshape_12/Reshape:output:0*
T0*/
_output_shapes
:���������2
decoder/softmax_2/Softmax�
decoder/softmax_1/SoftmaxSoftmax#decoder/reshape_11/Reshape:output:0*
T0*/
_output_shapes
:���������
2
decoder/softmax_1/Softmax�
IdentityIdentity#decoder/reshape_13/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity�

Identity_1Identity#decoder/reshape_14/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_1�

Identity_2Identity#decoder/softmax_1/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity_2�

Identity_3Identity#decoder/softmax_2/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������2

Identity_3�
NoOpNoOp(^decoder/dense_23/BiasAdd/ReadVariableOp'^decoder/dense_23/MatMul/ReadVariableOp(^decoder/dense_26/BiasAdd/ReadVariableOp'^decoder/dense_26/MatMul/ReadVariableOp(^decoder/dense_27/BiasAdd/ReadVariableOp'^decoder/dense_27/MatMul/ReadVariableOp(^decoder/dense_28/BiasAdd/ReadVariableOp'^decoder/dense_28/MatMul/ReadVariableOp(^decoder/dense_29/BiasAdd/ReadVariableOp'^decoder/dense_29/MatMul/ReadVariableOp(^decoder/dense_30/BiasAdd/ReadVariableOp'^decoder/dense_30/MatMul/ReadVariableOp(^decoder/dense_31/BiasAdd/ReadVariableOp'^decoder/dense_31/MatMul/ReadVariableOp(^decoder/dense_32/BiasAdd/ReadVariableOp'^decoder/dense_32/MatMul/ReadVariableOp(^decoder/dense_33/BiasAdd/ReadVariableOp'^decoder/dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 2R
'decoder/dense_23/BiasAdd/ReadVariableOp'decoder/dense_23/BiasAdd/ReadVariableOp2P
&decoder/dense_23/MatMul/ReadVariableOp&decoder/dense_23/MatMul/ReadVariableOp2R
'decoder/dense_26/BiasAdd/ReadVariableOp'decoder/dense_26/BiasAdd/ReadVariableOp2P
&decoder/dense_26/MatMul/ReadVariableOp&decoder/dense_26/MatMul/ReadVariableOp2R
'decoder/dense_27/BiasAdd/ReadVariableOp'decoder/dense_27/BiasAdd/ReadVariableOp2P
&decoder/dense_27/MatMul/ReadVariableOp&decoder/dense_27/MatMul/ReadVariableOp2R
'decoder/dense_28/BiasAdd/ReadVariableOp'decoder/dense_28/BiasAdd/ReadVariableOp2P
&decoder/dense_28/MatMul/ReadVariableOp&decoder/dense_28/MatMul/ReadVariableOp2R
'decoder/dense_29/BiasAdd/ReadVariableOp'decoder/dense_29/BiasAdd/ReadVariableOp2P
&decoder/dense_29/MatMul/ReadVariableOp&decoder/dense_29/MatMul/ReadVariableOp2R
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
:���������@
!
_user_specified_name	input_6
�
�
E__inference_dense_27_layer_call_and_return_conditional_losses_6736152

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�N
�
D__inference_decoder_layer_call_and_return_conditional_losses_6736778
input_6"
dense_23_6736721:@@
dense_23_6736723:@"
dense_32_6736726:@
dense_32_6736728:"
dense_30_6736731:@
dense_30_6736733:"
dense_28_6736736:@
dense_28_6736738:"
dense_26_6736741:@
dense_26_6736743:#
dense_33_6736746:	�N
dense_33_6736748:	�N#
dense_31_6736751:	�N
dense_31_6736753:	�N"
dense_29_6736756:0
dense_29_6736758:0"
dense_27_6736761:d
dense_27_6736763:d
identity

identity_1

identity_2

identity_3�� dense_23/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall� dense_30/StatefulPartitionedCall� dense_31/StatefulPartitionedCall� dense_32/StatefulPartitionedCall� dense_33/StatefulPartitionedCall�
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_23_6736721dense_23_6736723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_67360182"
 dense_23/StatefulPartitionedCall�
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_32_6736726dense_32_6736728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_67360352"
 dense_32/StatefulPartitionedCall�
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_30_6736731dense_30_6736733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_67360522"
 dense_30/StatefulPartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_28_6736736dense_28_6736738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_67360692"
 dense_28/StatefulPartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_26_6736741dense_26_6736743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_67360862"
 dense_26/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_6736746dense_33_6736748*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_67361022"
 dense_33/StatefulPartitionedCall�
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_6736751dense_31_6736753*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_67361182"
 dense_31/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_6736756dense_29_6736758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_67361352"
 dense_29/StatefulPartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_6736761dense_27_6736763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_67361522"
 dense_27/StatefulPartitionedCall�
re_lu_1/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_67361722
re_lu_1/PartitionedCall�
re_lu/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_67361882
re_lu/PartitionedCall�
reshape_12/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_12_layer_call_and_return_conditional_losses_67362042
reshape_12/PartitionedCall�
reshape_11/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_11_layer_call_and_return_conditional_losses_67362202
reshape_11/PartitionedCall�
reshape_14/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_14_layer_call_and_return_conditional_losses_67362372
reshape_14/PartitionedCall�
reshape_13/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_67362542
reshape_13/PartitionedCall�
softmax_2/PartitionedCallPartitionedCall#reshape_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_2_layer_call_and_return_conditional_losses_67362612
softmax_2/PartitionedCall�
softmax_1/PartitionedCallPartitionedCall#reshape_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_1_layer_call_and_return_conditional_losses_67362682
softmax_1/PartitionedCall�
IdentityIdentity"softmax_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity"softmax_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identity#reshape_13/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identity#reshape_14/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3�
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_6
�

`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6736172

inputs
identityL
NegNeginputs*
T0*(
_output_shapes
:����������N2
NegP
ReluReluNeg:y:0*
T0*(
_output_shapes
:����������N2
ReluS
Relu_1Reluinputs*
T0*(
_output_shapes
:����������N2
Relu_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1�
clip_by_value/MinimumMinimumRelu_1:activations:0Const:output:0*
T0*(
_output_shapes
:����������N2
clip_by_value/Minimum�
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*(
_output_shapes
:����������N2
clip_by_valueW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
Const_2j
mulMulConst_2:output:0Relu:activations:0*
T0*(
_output_shapes
:����������N2
mul`
subSubclip_by_value:z:0mul:z:0*
T0*(
_output_shapes
:����������N2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:����������N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�
�
E__inference_dense_29_layer_call_and_return_conditional_losses_6736135

inputs0
matmul_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������02
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_6736827
input_6
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:	�N

unknown_10:	�N

unknown_11:	�N

unknown_12:	�N

unknown_13:0

unknown_14:0

unknown_15:d

unknown_16:d
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *�
_output_shapesv
t:���������d:���������d:���������
:���������*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_67360002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:���������
2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*/
_output_shapes
:���������2

Identity_3h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_6
�N
�

#__inference__traced_restore_6737643
file_prefix2
 assignvariableop_dense_23_kernel:@@.
 assignvariableop_1_dense_23_bias:@4
"assignvariableop_2_dense_26_kernel:@.
 assignvariableop_3_dense_26_bias:4
"assignvariableop_4_dense_28_kernel:@.
 assignvariableop_5_dense_28_bias:4
"assignvariableop_6_dense_30_kernel:@.
 assignvariableop_7_dense_30_bias:4
"assignvariableop_8_dense_32_kernel:@.
 assignvariableop_9_dense_32_bias:5
#assignvariableop_10_dense_27_kernel:d/
!assignvariableop_11_dense_27_bias:d5
#assignvariableop_12_dense_29_kernel:0/
!assignvariableop_13_dense_29_bias:06
#assignvariableop_14_dense_31_kernel:	�N0
!assignvariableop_15_dense_31_bias:	�N6
#assignvariableop_16_dense_33_kernel:	�N0
!assignvariableop_17_dense_33_bias:	�N
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp assignvariableop_dense_23_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_23_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_26_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_26_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_28_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_28_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_30_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_30_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_32_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_32_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_27_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_27_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_29_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_29_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_31_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_31_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_33_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_33_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18f
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_19�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
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
�
�
E__inference_dense_23_layer_call_and_return_conditional_losses_6737196

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
��
�
D__inference_decoder_layer_call_and_return_conditional_losses_6736959

inputs9
'dense_23_matmul_readvariableop_resource:@@6
(dense_23_biasadd_readvariableop_resource:@9
'dense_32_matmul_readvariableop_resource:@6
(dense_32_biasadd_readvariableop_resource:9
'dense_30_matmul_readvariableop_resource:@6
(dense_30_biasadd_readvariableop_resource:9
'dense_28_matmul_readvariableop_resource:@6
(dense_28_biasadd_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource:@6
(dense_26_biasadd_readvariableop_resource::
'dense_33_matmul_readvariableop_resource:	�N7
(dense_33_biasadd_readvariableop_resource:	�N:
'dense_31_matmul_readvariableop_resource:	�N7
(dense_31_biasadd_readvariableop_resource:	�N9
'dense_29_matmul_readvariableop_resource:06
(dense_29_biasadd_readvariableop_resource:09
'dense_27_matmul_readvariableop_resource:d6
(dense_27_biasadd_readvariableop_resource:d
identity

identity_1

identity_2

identity_3��dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp�dense_26/BiasAdd/ReadVariableOp�dense_26/MatMul/ReadVariableOp�dense_27/BiasAdd/ReadVariableOp�dense_27/MatMul/ReadVariableOp�dense_28/BiasAdd/ReadVariableOp�dense_28/MatMul/ReadVariableOp�dense_29/BiasAdd/ReadVariableOp�dense_29/MatMul/ReadVariableOp�dense_30/BiasAdd/ReadVariableOp�dense_30/MatMul/ReadVariableOp�dense_31/BiasAdd/ReadVariableOp�dense_31/MatMul/ReadVariableOp�dense_32/BiasAdd/ReadVariableOp�dense_32/MatMul/ReadVariableOp�dense_33/BiasAdd/ReadVariableOp�dense_33/MatMul/ReadVariableOp�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02 
dense_23/MatMul/ReadVariableOp�
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_23/MatMul�
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_23/BiasAdd/ReadVariableOp�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_23/BiasAdds
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_23/Relu�
dense_32/MatMul/ReadVariableOpReadVariableOp'dense_32_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_32/MatMul/ReadVariableOp�
dense_32/MatMulMatMuldense_23/Relu:activations:0&dense_32/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_32/MatMul�
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_32/BiasAdd/ReadVariableOp�
dense_32/BiasAddBiasAdddense_32/MatMul:product:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_32/BiasAdds
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_32/Relu�
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_30/MatMul/ReadVariableOp�
dense_30/MatMulMatMuldense_23/Relu:activations:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_30/MatMul�
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_30/BiasAdd/ReadVariableOp�
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_30/Relu�
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_28/MatMul/ReadVariableOp�
dense_28/MatMulMatMuldense_23/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_28/MatMul�
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_28/BiasAdd/ReadVariableOp�
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_28/Relu�
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_26/MatMul/ReadVariableOp�
dense_26/MatMulMatMuldense_23/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/MatMul�
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOp�
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_26/Relu�
dense_33/MatMul/ReadVariableOpReadVariableOp'dense_33_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02 
dense_33/MatMul/ReadVariableOp�
dense_33/MatMulMatMuldense_32/Relu:activations:0&dense_33/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
dense_33/MatMul�
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02!
dense_33/BiasAdd/ReadVariableOp�
dense_33/BiasAddBiasAdddense_33/MatMul:product:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
dense_33/BiasAdd�
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02 
dense_31/MatMul/ReadVariableOp�
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
dense_31/MatMul�
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02!
dense_31/BiasAdd/ReadVariableOp�
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
dense_31/BiasAdd�
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02 
dense_29/MatMul/ReadVariableOp�
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02
dense_29/MatMul�
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_29/BiasAdd/ReadVariableOp�
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02
dense_29/BiasAdds
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*'
_output_shapes
:���������02
dense_29/Relu�
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_27/MatMul/ReadVariableOp�
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_27/MatMul�
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_27/BiasAdd/ReadVariableOp�
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
dense_27/Reluo
re_lu_1/NegNegdense_33/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
re_lu_1/Negh
re_lu_1/ReluRelure_lu_1/Neg:y:0*
T0*(
_output_shapes
:����������N2
re_lu_1/Reluv
re_lu_1/Relu_1Reludense_33/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
re_lu_1/Relu_1c
re_lu_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
re_lu_1/Constg
re_lu_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
re_lu_1/Const_1�
re_lu_1/clip_by_value/MinimumMinimumre_lu_1/Relu_1:activations:0re_lu_1/Const:output:0*
T0*(
_output_shapes
:����������N2
re_lu_1/clip_by_value/Minimum�
re_lu_1/clip_by_valueMaximum!re_lu_1/clip_by_value/Minimum:z:0re_lu_1/Const_1:output:0*
T0*(
_output_shapes
:����������N2
re_lu_1/clip_by_valueg
re_lu_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2
re_lu_1/Const_2�
re_lu_1/mulMulre_lu_1/Const_2:output:0re_lu_1/Relu:activations:0*
T0*(
_output_shapes
:����������N2
re_lu_1/mul�
re_lu_1/subSubre_lu_1/clip_by_value:z:0re_lu_1/mul:z:0*
T0*(
_output_shapes
:����������N2
re_lu_1/subk
	re_lu/NegNegdense_31/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
	re_lu/Negb

re_lu/ReluRelure_lu/Neg:y:0*
T0*(
_output_shapes
:����������N2

re_lu/Relur
re_lu/Relu_1Reludense_31/BiasAdd:output:0*
T0*(
_output_shapes
:����������N2
re_lu/Relu_1_
re_lu/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
re_lu/Constc
re_lu/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
re_lu/Const_1�
re_lu/clip_by_value/MinimumMinimumre_lu/Relu_1:activations:0re_lu/Const:output:0*
T0*(
_output_shapes
:����������N2
re_lu/clip_by_value/Minimum�
re_lu/clip_by_valueMaximumre_lu/clip_by_value/Minimum:z:0re_lu/Const_1:output:0*
T0*(
_output_shapes
:����������N2
re_lu/clip_by_valuec
re_lu/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2
re_lu/Const_2�
	re_lu/mulMulre_lu/Const_2:output:0re_lu/Relu:activations:0*
T0*(
_output_shapes
:����������N2
	re_lu/mulx
	re_lu/subSubre_lu/clip_by_value:z:0re_lu/mul:z:0*
T0*(
_output_shapes
:����������N2
	re_lu/subo
reshape_12/ShapeShapedense_29/Relu:activations:0*
T0*
_output_shapes
:2
reshape_12/Shape�
reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_12/strided_slice/stack�
 reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_1�
 reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_2�
reshape_12/strided_sliceStridedSlicereshape_12/Shape:output:0'reshape_12/strided_slice/stack:output:0)reshape_12/strided_slice/stack_1:output:0)reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_12/strided_slicez
reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/1z
reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/2z
reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_12/Reshape/shape/3�
reshape_12/Reshape/shapePack!reshape_12/strided_slice:output:0#reshape_12/Reshape/shape/1:output:0#reshape_12/Reshape/shape/2:output:0#reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_12/Reshape/shape�
reshape_12/ReshapeReshapedense_29/Relu:activations:0!reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
reshape_12/Reshapeo
reshape_11/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
reshape_11/Shape�
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_11/strided_slice/stack�
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_1�
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_2�
reshape_11/strided_sliceStridedSlicereshape_11/Shape:output:0'reshape_11/strided_slice/stack:output:0)reshape_11/strided_slice/stack_1:output:0)reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_11/strided_slicez
reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/1z
reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_11/Reshape/shape/2z
reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape_11/Reshape/shape/3�
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0#reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_11/Reshape/shape�
reshape_11/ReshapeReshapedense_27/Relu:activations:0!reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:���������
2
reshape_11/Reshapec
reshape_14/ShapeShapere_lu_1/sub:z:0*
T0*
_output_shapes
:2
reshape_14/Shape�
reshape_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_14/strided_slice/stack�
 reshape_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_1�
 reshape_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_14/strided_slice/stack_2�
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
value	B :d2
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
reshape_14/Reshape/shape/4�
reshape_14/Reshape/shapePack!reshape_14/strided_slice:output:0#reshape_14/Reshape/shape/1:output:0#reshape_14/Reshape/shape/2:output:0#reshape_14/Reshape/shape/3:output:0#reshape_14/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape_14/Reshape/shape�
reshape_14/ReshapeReshapere_lu_1/sub:z:0!reshape_14/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������d2
reshape_14/Reshapea
reshape_13/ShapeShapere_lu/sub:z:0*
T0*
_output_shapes
:2
reshape_13/Shape�
reshape_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_13/strided_slice/stack�
 reshape_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_1�
 reshape_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_13/strided_slice/stack_2�
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
value	B :d2
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
reshape_13/Reshape/shape/4�
reshape_13/Reshape/shapePack!reshape_13/strided_slice:output:0#reshape_13/Reshape/shape/1:output:0#reshape_13/Reshape/shape/2:output:0#reshape_13/Reshape/shape/3:output:0#reshape_13/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape_13/Reshape/shape�
reshape_13/ReshapeReshapere_lu/sub:z:0!reshape_13/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������d2
reshape_13/Reshape�
softmax_2/SoftmaxSoftmaxreshape_12/Reshape:output:0*
T0*/
_output_shapes
:���������2
softmax_2/Softmax�
softmax_1/SoftmaxSoftmaxreshape_11/Reshape:output:0*
T0*/
_output_shapes
:���������
2
softmax_1/Softmax~
IdentityIdentitysoftmax_1/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identitysoftmax_2/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identityreshape_13/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identityreshape_14/Reshape:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3�
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp ^dense_32/BiasAdd/ReadVariableOp^dense_32/MatMul/ReadVariableOp ^dense_33/BiasAdd/ReadVariableOp^dense_33/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp2B
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
:���������@
 
_user_specified_nameinputs
�
c
G__inference_reshape_11_layer_call_and_return_conditional_losses_6736220

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
strided_slice/stack_2�
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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������
2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
E__inference_dense_26_layer_call_and_return_conditional_losses_6737216

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_dense_23_layer_call_and_return_conditional_losses_6736018

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
C
'__inference_re_lu_layer_call_fn_6737420

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_67361882
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�

^
B__inference_re_lu_layer_call_and_return_conditional_losses_6737415

inputs
identityL
NegNeginputs*
T0*(
_output_shapes
:����������N2
NegP
ReluReluNeg:y:0*
T0*(
_output_shapes
:����������N2
ReluS
Relu_1Reluinputs*
T0*(
_output_shapes
:����������N2
Relu_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1�
clip_by_value/MinimumMinimumRelu_1:activations:0Const:output:0*
T0*(
_output_shapes
:����������N2
clip_by_value/Minimum�
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*(
_output_shapes
:����������N2
clip_by_valueW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
Const_2j
mulMulConst_2:output:0Relu:activations:0*
T0*(
_output_shapes
:����������N2
mul`
subSubclip_by_value:z:0mul:z:0*
T0*(
_output_shapes
:����������N2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:����������N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�
c
G__inference_reshape_14_layer_call_and_return_conditional_losses_6736237

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
strided_slice/stack_2�
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
value	B :d2
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
Reshape/shape/4�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:���������d2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�
�
E__inference_dense_30_layer_call_and_return_conditional_losses_6736052

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_6737185

inputs
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:	�N

unknown_10:	�N

unknown_11:	�N

unknown_12:	�N

unknown_13:0

unknown_14:0

unknown_15:d

unknown_16:d
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *�
_output_shapesv
t:���������
:���������:���������d:���������d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_67365662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_dense_26_layer_call_fn_6737225

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_67360862
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�N
�
D__inference_decoder_layer_call_and_return_conditional_losses_6736274

inputs"
dense_23_6736019:@@
dense_23_6736021:@"
dense_32_6736036:@
dense_32_6736038:"
dense_30_6736053:@
dense_30_6736055:"
dense_28_6736070:@
dense_28_6736072:"
dense_26_6736087:@
dense_26_6736089:#
dense_33_6736103:	�N
dense_33_6736105:	�N#
dense_31_6736119:	�N
dense_31_6736121:	�N"
dense_29_6736136:0
dense_29_6736138:0"
dense_27_6736153:d
dense_27_6736155:d
identity

identity_1

identity_2

identity_3�� dense_23/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall� dense_30/StatefulPartitionedCall� dense_31/StatefulPartitionedCall� dense_32/StatefulPartitionedCall� dense_33/StatefulPartitionedCall�
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_6736019dense_23_6736021*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_67360182"
 dense_23/StatefulPartitionedCall�
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_32_6736036dense_32_6736038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_67360352"
 dense_32/StatefulPartitionedCall�
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_30_6736053dense_30_6736055*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_67360522"
 dense_30/StatefulPartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_28_6736070dense_28_6736072*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_67360692"
 dense_28/StatefulPartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_26_6736087dense_26_6736089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_67360862"
 dense_26/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_6736103dense_33_6736105*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_67361022"
 dense_33/StatefulPartitionedCall�
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_6736119dense_31_6736121*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_67361182"
 dense_31/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_6736136dense_29_6736138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_67361352"
 dense_29/StatefulPartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_6736153dense_27_6736155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_67361522"
 dense_27/StatefulPartitionedCall�
re_lu_1/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_67361722
re_lu_1/PartitionedCall�
re_lu/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_67361882
re_lu/PartitionedCall�
reshape_12/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_12_layer_call_and_return_conditional_losses_67362042
reshape_12/PartitionedCall�
reshape_11/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_11_layer_call_and_return_conditional_losses_67362202
reshape_11/PartitionedCall�
reshape_14/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_14_layer_call_and_return_conditional_losses_67362372
reshape_14/PartitionedCall�
reshape_13/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_67362542
reshape_13/PartitionedCall�
softmax_2/PartitionedCallPartitionedCall#reshape_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_2_layer_call_and_return_conditional_losses_67362612
softmax_2/PartitionedCall�
softmax_1/PartitionedCallPartitionedCall#reshape_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_1_layer_call_and_return_conditional_losses_67362682
softmax_1/PartitionedCall�
IdentityIdentity"softmax_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity"softmax_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identity#reshape_13/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identity#reshape_14/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3�
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
G
+__inference_softmax_2_layer_call_fn_6737459

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_2_layer_call_and_return_conditional_losses_67362612
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_28_layer_call_and_return_conditional_losses_6737236

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
G__inference_reshape_13_layer_call_and_return_conditional_losses_6737474

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
strided_slice/stack_2�
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
value	B :d2
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
Reshape/shape/4�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:���������d2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�

^
B__inference_re_lu_layer_call_and_return_conditional_losses_6736188

inputs
identityL
NegNeginputs*
T0*(
_output_shapes
:����������N2
NegP
ReluReluNeg:y:0*
T0*(
_output_shapes
:����������N2
ReluS
Relu_1Reluinputs*
T0*(
_output_shapes
:����������N2
Relu_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1�
clip_by_value/MinimumMinimumRelu_1:activations:0Const:output:0*
T0*(
_output_shapes
:����������N2
clip_by_value/Minimum�
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*(
_output_shapes
:����������N2
clip_by_valueW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
Const_2j
mulMulConst_2:output:0Relu:activations:0*
T0*(
_output_shapes
:����������N2
mul`
subSubclip_by_value:z:0mul:z:0*
T0*(
_output_shapes
:����������N2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:����������N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�
c
G__inference_reshape_14_layer_call_and_return_conditional_losses_6737494

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
strided_slice/stack_2�
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
value	B :d2
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
Reshape/shape/4�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:���������d2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�

`
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6737434

inputs
identityL
NegNeginputs*
T0*(
_output_shapes
:����������N2
NegP
ReluReluNeg:y:0*
T0*(
_output_shapes
:����������N2
ReluS
Relu_1Reluinputs*
T0*(
_output_shapes
:����������N2
Relu_1S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2	
Const_1�
clip_by_value/MinimumMinimumRelu_1:activations:0Const:output:0*
T0*(
_output_shapes
:����������N2
clip_by_value/Minimum�
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_1:output:0*
T0*(
_output_shapes
:����������N2
clip_by_valueW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
Const_2j
mulMulConst_2:output:0Relu:activations:0*
T0*(
_output_shapes
:����������N2
mul`
subSubclip_by_value:z:0mul:z:0*
T0*(
_output_shapes
:����������N2
sub\
IdentityIdentitysub:z:0*
T0*(
_output_shapes
:����������N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_6736319
input_6
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:	�N

unknown_10:	�N

unknown_11:	�N

unknown_12:	�N

unknown_13:0

unknown_14:0

unknown_15:d

unknown_16:d
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *�
_output_shapesv
t:���������
:���������:���������d:���������d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_67362742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_6
�
�
)__inference_decoder_layer_call_fn_6736658
input_6
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:	�N

unknown_10:	�N

unknown_11:	�N

unknown_12:	�N

unknown_13:0

unknown_14:0

unknown_15:d

unknown_16:d
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *�
_output_shapesv
t:���������
:���������:���������d:���������d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_67365662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_6
�
c
G__inference_reshape_12_layer_call_and_return_conditional_losses_6737396

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
strided_slice/stack_2�
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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������0:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
*__inference_dense_27_layer_call_fn_6737305

inputs
unknown:d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_67361522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_reshape_12_layer_call_fn_6737401

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_12_layer_call_and_return_conditional_losses_67362042
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������0:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
E
)__inference_re_lu_1_layer_call_fn_6737439

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_67361722
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������N2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�N
�
D__inference_decoder_layer_call_and_return_conditional_losses_6736718
input_6"
dense_23_6736661:@@
dense_23_6736663:@"
dense_32_6736666:@
dense_32_6736668:"
dense_30_6736671:@
dense_30_6736673:"
dense_28_6736676:@
dense_28_6736678:"
dense_26_6736681:@
dense_26_6736683:#
dense_33_6736686:	�N
dense_33_6736688:	�N#
dense_31_6736691:	�N
dense_31_6736693:	�N"
dense_29_6736696:0
dense_29_6736698:0"
dense_27_6736701:d
dense_27_6736703:d
identity

identity_1

identity_2

identity_3�� dense_23/StatefulPartitionedCall� dense_26/StatefulPartitionedCall� dense_27/StatefulPartitionedCall� dense_28/StatefulPartitionedCall� dense_29/StatefulPartitionedCall� dense_30/StatefulPartitionedCall� dense_31/StatefulPartitionedCall� dense_32/StatefulPartitionedCall� dense_33/StatefulPartitionedCall�
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_23_6736661dense_23_6736663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_67360182"
 dense_23/StatefulPartitionedCall�
 dense_32/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_32_6736666dense_32_6736668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_67360352"
 dense_32/StatefulPartitionedCall�
 dense_30/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_30_6736671dense_30_6736673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_67360522"
 dense_30/StatefulPartitionedCall�
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_28_6736676dense_28_6736678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_67360692"
 dense_28/StatefulPartitionedCall�
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_26_6736681dense_26_6736683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_26_layer_call_and_return_conditional_losses_67360862"
 dense_26/StatefulPartitionedCall�
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_6736686dense_33_6736688*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_67361022"
 dense_33/StatefulPartitionedCall�
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_6736691dense_31_6736693*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_67361182"
 dense_31/StatefulPartitionedCall�
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_6736696dense_29_6736698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_67361352"
 dense_29/StatefulPartitionedCall�
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_6736701dense_27_6736703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_27_layer_call_and_return_conditional_losses_67361522"
 dense_27/StatefulPartitionedCall�
re_lu_1/PartitionedCallPartitionedCall)dense_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_re_lu_1_layer_call_and_return_conditional_losses_67361722
re_lu_1/PartitionedCall�
re_lu/PartitionedCallPartitionedCall)dense_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_re_lu_layer_call_and_return_conditional_losses_67361882
re_lu/PartitionedCall�
reshape_12/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_12_layer_call_and_return_conditional_losses_67362042
reshape_12/PartitionedCall�
reshape_11/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_11_layer_call_and_return_conditional_losses_67362202
reshape_11/PartitionedCall�
reshape_14/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_14_layer_call_and_return_conditional_losses_67362372
reshape_14/PartitionedCall�
reshape_13/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_13_layer_call_and_return_conditional_losses_67362542
reshape_13/PartitionedCall�
softmax_2/PartitionedCallPartitionedCall#reshape_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_2_layer_call_and_return_conditional_losses_67362612
softmax_2/PartitionedCall�
softmax_1/PartitionedCallPartitionedCall#reshape_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_1_layer_call_and_return_conditional_losses_67362682
softmax_1/PartitionedCall�
IdentityIdentity"softmax_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity"softmax_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identity#reshape_13/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identity#reshape_14/PartitionedCall:output:0^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3�
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:P L
'
_output_shapes
:���������@
!
_user_specified_name	input_6
�
�
*__inference_dense_23_layer_call_fn_6737205

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_67360182
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_dense_26_layer_call_and_return_conditional_losses_6736086

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
G
+__inference_softmax_1_layer_call_fn_6737449

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_softmax_1_layer_call_and_return_conditional_losses_67362682
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
*__inference_dense_30_layer_call_fn_6737265

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_30_layer_call_and_return_conditional_losses_67360522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
c
G__inference_reshape_11_layer_call_and_return_conditional_losses_6737377

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
strided_slice/stack_2�
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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������
2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
H
,__inference_reshape_14_layer_call_fn_6737499

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:���������d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_14_layer_call_and_return_conditional_losses_67362372
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�
�
*__inference_dense_33_layer_call_fn_6737363

inputs
unknown:	�N
	unknown_0:	�N
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_33_layer_call_and_return_conditional_losses_67361022
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������N2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_reshape_13_layer_call_and_return_conditional_losses_6736254

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
strided_slice/stack_2�
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
value	B :d2
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
Reshape/shape/4�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:���������d2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:���������d2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������N:P L
(
_output_shapes
:����������N
 
_user_specified_nameinputs
�
b
F__inference_softmax_2_layer_call_and_return_conditional_losses_6736261

inputs
identity_
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:���������2	
Softmaxm
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_dense_27_layer_call_and_return_conditional_losses_6737296

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������d2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
b
F__inference_softmax_2_layer_call_and_return_conditional_losses_6737454

inputs
identity_
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:���������2	
Softmaxm
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_decoder_layer_call_fn_6737138

inputs
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:
	unknown_3:@
	unknown_4:
	unknown_5:@
	unknown_6:
	unknown_7:@
	unknown_8:
	unknown_9:	�N

unknown_10:	�N

unknown_11:	�N

unknown_12:	�N

unknown_13:0

unknown_14:0

unknown_15:d

unknown_16:d
identity

identity_1

identity_2

identity_3��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *�
_output_shapesv
t:���������
:���������:���������d:���������d*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_67362742
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������
2

Identity�

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:���������2

Identity_1�

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_2�

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*3
_output_shapes!
:���������d2

Identity_3h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_dense_28_layer_call_fn_6737245

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_67360692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_dense_32_layer_call_and_return_conditional_losses_6736035

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
*__inference_dense_31_layer_call_fn_6737344

inputs
unknown:	�N
	unknown_0:	�N
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������N*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_31_layer_call_and_return_conditional_losses_67361182
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������N2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_31_layer_call_and_return_conditional_losses_6737335

inputs1
matmul_readvariableop_resource:	�N.
biasadd_readvariableop_resource:	�N
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������N2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_reshape_11_layer_call_fn_6737382

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_reshape_11_layer_call_and_return_conditional_losses_67362202
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������d:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
E__inference_dense_33_layer_call_and_return_conditional_losses_6736102

inputs1
matmul_readvariableop_resource:	�N.
biasadd_readvariableop_resource:	�N
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������N2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
E__inference_dense_33_layer_call_and_return_conditional_losses_6737354

inputs1
matmul_readvariableop_resource:	�N.
biasadd_readvariableop_resource:	�N
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�N*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�N*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������N2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������N2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_dense_32_layer_call_fn_6737285

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dense_32_layer_call_and_return_conditional_losses_67360352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_dense_29_layer_call_and_return_conditional_losses_6737316

inputs0
matmul_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������02	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������02
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_reshape_12_layer_call_and_return_conditional_losses_6736204

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
strided_slice/stack_2�
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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������0:O K
'
_output_shapes
:���������0
 
_user_specified_nameinputs
�
�
E__inference_dense_32_layer_call_and_return_conditional_losses_6737276

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_60
serving_default_input_6:0���������@J

reshape_13<
StatefulPartitionedCall:0���������dJ

reshape_14<
StatefulPartitionedCall:1���������dE
	softmax_18
StatefulPartitionedCall:2���������
E
	softmax_28
StatefulPartitionedCall:3���������tensorflow/serving/predict:��
�
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
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

%kernel
&bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

1kernel
2bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

7kernel
8bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

=kernel
>bias
?regularization_losses
@trainable_variables
A	variables
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Ikernel
Jbias
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
[regularization_losses
\trainable_variables
]	variables
^	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
_regularization_losses
`trainable_variables
a	variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
gregularization_losses
htrainable_variables
i	variables
j	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
 "
trackable_list_wrapper
�
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
I16
J17"
trackable_list_wrapper
�
0
1
2
 3
%4
&5
+6
,7
18
29
710
811
=12
>13
C14
D15
I16
J17"
trackable_list_wrapper
�
olayer_metrics
player_regularization_losses

qlayers
regularization_losses
rnon_trainable_variables
smetrics
	variables
trainable_variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
!:@@2dense_23/kernel
:@2dense_23/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
tlayer_metrics

ulayers
vnon_trainable_variables
regularization_losses
wmetrics
trainable_variables
xlayer_regularization_losses
	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_26/kernel
:2dense_26/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
�
ylayer_metrics

zlayers
{non_trainable_variables
!regularization_losses
|metrics
"trainable_variables
}layer_regularization_losses
#	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_28/kernel
:2dense_28/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
�
~layer_metrics

layers
�non_trainable_variables
'regularization_losses
�metrics
(trainable_variables
 �layer_regularization_losses
)	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_30/kernel
:2dense_30/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
-regularization_losses
�metrics
.trainable_variables
 �layer_regularization_losses
/	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_32/kernel
:2dense_32/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
3regularization_losses
�metrics
4trainable_variables
 �layer_regularization_losses
5	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:d2dense_27/kernel
:d2dense_27/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
9regularization_losses
�metrics
:trainable_variables
 �layer_regularization_losses
;	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:02dense_29/kernel
:02dense_29/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
?regularization_losses
�metrics
@trainable_variables
 �layer_regularization_losses
A	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�N2dense_31/kernel
:�N2dense_31/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
Eregularization_losses
�metrics
Ftrainable_variables
 �layer_regularization_losses
G	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 	�N2dense_33/kernel
:�N2dense_33/bias
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
Kregularization_losses
�metrics
Ltrainable_variables
 �layer_regularization_losses
M	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
Oregularization_losses
�metrics
Ptrainable_variables
 �layer_regularization_losses
Q	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
Sregularization_losses
�metrics
Ttrainable_variables
 �layer_regularization_losses
U	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
Wregularization_losses
�metrics
Xtrainable_variables
 �layer_regularization_losses
Y	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
[regularization_losses
�metrics
\trainable_variables
 �layer_regularization_losses
]	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
_regularization_losses
�metrics
`trainable_variables
 �layer_regularization_losses
a	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
cregularization_losses
�metrics
dtrainable_variables
 �layer_regularization_losses
e	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
gregularization_losses
�metrics
htrainable_variables
 �layer_regularization_losses
i	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�layer_metrics
�layers
�non_trainable_variables
kregularization_losses
�metrics
ltrainable_variables
 �layer_regularization_losses
m	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
�
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
14
15
16
17"
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
�2�
D__inference_decoder_layer_call_and_return_conditional_losses_6736959
D__inference_decoder_layer_call_and_return_conditional_losses_6737091
D__inference_decoder_layer_call_and_return_conditional_losses_6736718
D__inference_decoder_layer_call_and_return_conditional_losses_6736778�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_6736000input_6"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_decoder_layer_call_fn_6736319
)__inference_decoder_layer_call_fn_6737138
)__inference_decoder_layer_call_fn_6737185
)__inference_decoder_layer_call_fn_6736658�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_dense_23_layer_call_and_return_conditional_losses_6737196�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_23_layer_call_fn_6737205�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_26_layer_call_and_return_conditional_losses_6737216�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_26_layer_call_fn_6737225�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_28_layer_call_and_return_conditional_losses_6737236�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_28_layer_call_fn_6737245�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_30_layer_call_and_return_conditional_losses_6737256�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_30_layer_call_fn_6737265�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_32_layer_call_and_return_conditional_losses_6737276�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_32_layer_call_fn_6737285�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_27_layer_call_and_return_conditional_losses_6737296�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_27_layer_call_fn_6737305�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_29_layer_call_and_return_conditional_losses_6737316�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_29_layer_call_fn_6737325�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_31_layer_call_and_return_conditional_losses_6737335�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_31_layer_call_fn_6737344�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_33_layer_call_and_return_conditional_losses_6737354�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_33_layer_call_fn_6737363�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_reshape_11_layer_call_and_return_conditional_losses_6737377�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_reshape_11_layer_call_fn_6737382�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_reshape_12_layer_call_and_return_conditional_losses_6737396�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_reshape_12_layer_call_fn_6737401�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_re_lu_layer_call_and_return_conditional_losses_6737415�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_re_lu_layer_call_fn_6737420�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6737434�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_re_lu_1_layer_call_fn_6737439�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_softmax_1_layer_call_and_return_conditional_losses_6737444�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_softmax_1_layer_call_fn_6737449�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_softmax_2_layer_call_and_return_conditional_losses_6737454�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_softmax_2_layer_call_fn_6737459�
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_reshape_13_layer_call_and_return_conditional_losses_6737474�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_reshape_13_layer_call_fn_6737479�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_reshape_14_layer_call_and_return_conditional_losses_6737494�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_reshape_14_layer_call_fn_6737499�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_6736827input_6"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_6736000�12+,%& IJCD=>780�-
&�#
!�
input_6���������@
� "���
>

reshape_130�-

reshape_13���������d
>

reshape_140�-

reshape_14���������d
8
	softmax_1+�(
	softmax_1���������

8
	softmax_2+�(
	softmax_2����������
D__inference_decoder_layer_call_and_return_conditional_losses_6736718�12+,%& IJCD=>788�5
.�+
!�
input_6���������@
p 

 
� "���
���
%�"
0/0���������

%�"
0/1���������
)�&
0/2���������d
)�&
0/3���������d
� �
D__inference_decoder_layer_call_and_return_conditional_losses_6736778�12+,%& IJCD=>788�5
.�+
!�
input_6���������@
p

 
� "���
���
%�"
0/0���������

%�"
0/1���������
)�&
0/2���������d
)�&
0/3���������d
� �
D__inference_decoder_layer_call_and_return_conditional_losses_6736959�12+,%& IJCD=>787�4
-�*
 �
inputs���������@
p 

 
� "���
���
%�"
0/0���������

%�"
0/1���������
)�&
0/2���������d
)�&
0/3���������d
� �
D__inference_decoder_layer_call_and_return_conditional_losses_6737091�12+,%& IJCD=>787�4
-�*
 �
inputs���������@
p

 
� "���
���
%�"
0/0���������

%�"
0/1���������
)�&
0/2���������d
)�&
0/3���������d
� �
)__inference_decoder_layer_call_fn_6736319�12+,%& IJCD=>788�5
.�+
!�
input_6���������@
p 

 
� "���
#� 
0���������

#� 
1���������
'�$
2���������d
'�$
3���������d�
)__inference_decoder_layer_call_fn_6736658�12+,%& IJCD=>788�5
.�+
!�
input_6���������@
p

 
� "���
#� 
0���������

#� 
1���������
'�$
2���������d
'�$
3���������d�
)__inference_decoder_layer_call_fn_6737138�12+,%& IJCD=>787�4
-�*
 �
inputs���������@
p 

 
� "���
#� 
0���������

#� 
1���������
'�$
2���������d
'�$
3���������d�
)__inference_decoder_layer_call_fn_6737185�12+,%& IJCD=>787�4
-�*
 �
inputs���������@
p

 
� "���
#� 
0���������

#� 
1���������
'�$
2���������d
'�$
3���������d�
E__inference_dense_23_layer_call_and_return_conditional_losses_6737196\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� }
*__inference_dense_23_layer_call_fn_6737205O/�,
%�"
 �
inputs���������@
� "����������@�
E__inference_dense_26_layer_call_and_return_conditional_losses_6737216\ /�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_26_layer_call_fn_6737225O /�,
%�"
 �
inputs���������@
� "�����������
E__inference_dense_27_layer_call_and_return_conditional_losses_6737296\78/�,
%�"
 �
inputs���������
� "%�"
�
0���������d
� }
*__inference_dense_27_layer_call_fn_6737305O78/�,
%�"
 �
inputs���������
� "����������d�
E__inference_dense_28_layer_call_and_return_conditional_losses_6737236\%&/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_28_layer_call_fn_6737245O%&/�,
%�"
 �
inputs���������@
� "�����������
E__inference_dense_29_layer_call_and_return_conditional_losses_6737316\=>/�,
%�"
 �
inputs���������
� "%�"
�
0���������0
� }
*__inference_dense_29_layer_call_fn_6737325O=>/�,
%�"
 �
inputs���������
� "����������0�
E__inference_dense_30_layer_call_and_return_conditional_losses_6737256\+,/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_30_layer_call_fn_6737265O+,/�,
%�"
 �
inputs���������@
� "�����������
E__inference_dense_31_layer_call_and_return_conditional_losses_6737335]CD/�,
%�"
 �
inputs���������
� "&�#
�
0����������N
� ~
*__inference_dense_31_layer_call_fn_6737344PCD/�,
%�"
 �
inputs���������
� "�����������N�
E__inference_dense_32_layer_call_and_return_conditional_losses_6737276\12/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_32_layer_call_fn_6737285O12/�,
%�"
 �
inputs���������@
� "�����������
E__inference_dense_33_layer_call_and_return_conditional_losses_6737354]IJ/�,
%�"
 �
inputs���������
� "&�#
�
0����������N
� ~
*__inference_dense_33_layer_call_fn_6737363PIJ/�,
%�"
 �
inputs���������
� "�����������N�
D__inference_re_lu_1_layer_call_and_return_conditional_losses_6737434Z0�-
&�#
!�
inputs����������N
� "&�#
�
0����������N
� z
)__inference_re_lu_1_layer_call_fn_6737439M0�-
&�#
!�
inputs����������N
� "�����������N�
B__inference_re_lu_layer_call_and_return_conditional_losses_6737415Z0�-
&�#
!�
inputs����������N
� "&�#
�
0����������N
� x
'__inference_re_lu_layer_call_fn_6737420M0�-
&�#
!�
inputs����������N
� "�����������N�
G__inference_reshape_11_layer_call_and_return_conditional_losses_6737377`/�,
%�"
 �
inputs���������d
� "-�*
#� 
0���������

� �
,__inference_reshape_11_layer_call_fn_6737382S/�,
%�"
 �
inputs���������d
� " ����������
�
G__inference_reshape_12_layer_call_and_return_conditional_losses_6737396`/�,
%�"
 �
inputs���������0
� "-�*
#� 
0���������
� �
,__inference_reshape_12_layer_call_fn_6737401S/�,
%�"
 �
inputs���������0
� " �����������
G__inference_reshape_13_layer_call_and_return_conditional_losses_6737474e0�-
&�#
!�
inputs����������N
� "1�.
'�$
0���������d
� �
,__inference_reshape_13_layer_call_fn_6737479X0�-
&�#
!�
inputs����������N
� "$�!���������d�
G__inference_reshape_14_layer_call_and_return_conditional_losses_6737494e0�-
&�#
!�
inputs����������N
� "1�.
'�$
0���������d
� �
,__inference_reshape_14_layer_call_fn_6737499X0�-
&�#
!�
inputs����������N
� "$�!���������d�
%__inference_signature_wrapper_6736827�12+,%& IJCD=>78;�8
� 
1�.
,
input_6!�
input_6���������@"���
>

reshape_130�-

reshape_13���������d
>

reshape_140�-

reshape_14���������d
8
	softmax_1+�(
	softmax_1���������

8
	softmax_2+�(
	softmax_2����������
F__inference_softmax_1_layer_call_and_return_conditional_losses_6737444l;�8
1�.
(�%
inputs���������


 
� "-�*
#� 
0���������

� �
+__inference_softmax_1_layer_call_fn_6737449_;�8
1�.
(�%
inputs���������


 
� " ����������
�
F__inference_softmax_2_layer_call_and_return_conditional_losses_6737454l;�8
1�.
(�%
inputs���������

 
� "-�*
#� 
0���������
� �
+__inference_softmax_2_layer_call_fn_6737459_;�8
1�.
(�%
inputs���������

 
� " ����������