я
Ђѓ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
О
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
і
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8Ћ
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:	*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0

encoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:0 *&
shared_nameencoder_output/kernel

)encoder_output/kernel/Read/ReadVariableOpReadVariableOpencoder_output/kernel*
_output_shapes

:0 *
dtype0
~
encoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameencoder_output/bias
w
'encoder_output/bias/Read/ReadVariableOpReadVariableOpencoder_output/bias*
_output_shapes
: *
dtype0
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
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_26/kernel
s
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel*
_output_shapes

: *
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
: * 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

: *
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

NoOpNoOp
Лg
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*іf
valueьfBщf Bтf
А
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
R
#trainable_variables
$regularization_losses
%	variables
&	keras_api
R
'trainable_variables
(regularization_losses
)	variables
*	keras_api
h

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
h

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
 
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
R
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

?kernel
@bias
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
h

Ekernel
Fbias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
h

Kkernel
Lbias
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
R
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
h

Ukernel
Vbias
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
Т
[layer-0
\layer_with_weights-0
\layer-1
]layer_with_weights-1
]layer-2
^layer_with_weights-2
^layer-3
_layer_with_weights-3
_layer-4
`layer_with_weights-4
`layer-5
alayer_with_weights-5
alayer-6
blayer_with_weights-6
blayer-7
clayer-8
dlayer-9
elayer-10
flayer-11
glayer-12
hlayer-13
itrainable_variables
jregularization_losses
k	variables
l	keras_api
ц
0
1
2
3
+4
,5
16
27
?8
@9
E10
F11
K12
L13
U14
V15
m16
n17
o18
p19
q20
r21
s22
t23
u24
v25
w26
x27
y28
z29
 
ц
0
1
2
3
+4
,5
16
27
?8
@9
E10
F11
K12
L13
U14
V15
m16
n17
o18
p19
q20
r21
s22
t23
u24
v25
w26
x27
y28
z29
­
trainable_variables
regularization_losses

{layers
	variables
|layer_regularization_losses
}metrics
~layer_metrics
non_trainable_variables
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
В
trainable_variables
regularization_losses
layers
 layer_regularization_losses
	variables
metrics
layer_metrics
non_trainable_variables
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
В
trainable_variables
 regularization_losses
layers
 layer_regularization_losses
!	variables
metrics
layer_metrics
non_trainable_variables
 
 
 
В
#trainable_variables
$regularization_losses
layers
 layer_regularization_losses
%	variables
metrics
layer_metrics
non_trainable_variables
 
 
 
В
'trainable_variables
(regularization_losses
layers
 layer_regularization_losses
)	variables
metrics
layer_metrics
non_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
В
-trainable_variables
.regularization_losses
layers
 layer_regularization_losses
/	variables
metrics
layer_metrics
non_trainable_variables
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
В
3trainable_variables
4regularization_losses
layers
 layer_regularization_losses
5	variables
metrics
layer_metrics
non_trainable_variables
 
 
 
В
7trainable_variables
8regularization_losses
layers
 layer_regularization_losses
9	variables
 metrics
Ёlayer_metrics
Ђnon_trainable_variables
 
 
 
В
;trainable_variables
<regularization_losses
Ѓlayers
 Єlayer_regularization_losses
=	variables
Ѕmetrics
Іlayer_metrics
Їnon_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

?0
@1
В
Atrainable_variables
Bregularization_losses
Јlayers
 Љlayer_regularization_losses
C	variables
Њmetrics
Ћlayer_metrics
Ќnon_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

E0
F1
 

E0
F1
В
Gtrainable_variables
Hregularization_losses
­layers
 Ўlayer_regularization_losses
I	variables
Џmetrics
Аlayer_metrics
Бnon_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
В
Mtrainable_variables
Nregularization_losses
Вlayers
 Гlayer_regularization_losses
O	variables
Дmetrics
Еlayer_metrics
Жnon_trainable_variables
 
 
 
В
Qtrainable_variables
Rregularization_losses
Зlayers
 Иlayer_regularization_losses
S	variables
Йmetrics
Кlayer_metrics
Лnon_trainable_variables
a_
VARIABLE_VALUEencoder_output/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEencoder_output/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

U0
V1
 

U0
V1
В
Wtrainable_variables
Xregularization_losses
Мlayers
 Нlayer_regularization_losses
Y	variables
Оmetrics
Пlayer_metrics
Рnon_trainable_variables

С_init_input_shape
l

mkernel
nbias
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
l

okernel
pbias
Цtrainable_variables
Чregularization_losses
Ш	variables
Щ	keras_api
l

qkernel
rbias
Ъtrainable_variables
Ыregularization_losses
Ь	variables
Э	keras_api
l

skernel
tbias
Юtrainable_variables
Яregularization_losses
а	variables
б	keras_api
l

ukernel
vbias
вtrainable_variables
гregularization_losses
д	variables
е	keras_api
l

wkernel
xbias
жtrainable_variables
зregularization_losses
и	variables
й	keras_api
l

ykernel
zbias
кtrainable_variables
лregularization_losses
м	variables
н	keras_api
V
оtrainable_variables
пregularization_losses
р	variables
с	keras_api
V
тtrainable_variables
уregularization_losses
ф	variables
х	keras_api
V
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
V
ъtrainable_variables
ыregularization_losses
ь	variables
э	keras_api
V
юtrainable_variables
яregularization_losses
№	variables
ё	keras_api
V
ђtrainable_variables
ѓregularization_losses
є	variables
ѕ	keras_api
f
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13
 
f
m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13
В
itrainable_variables
jregularization_losses
іlayers
k	variables
 їlayer_regularization_losses
јmetrics
љlayer_metrics
њnon_trainable_variables
VT
VARIABLE_VALUEdense_23/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_23/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_24/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_24/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_26/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_26/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_28/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_28/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_25/kernel1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_25/bias1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_27/kernel1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_27/bias1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_29/kernel1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_29/bias1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE
~
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

m0
n1
 

m0
n1
Е
Тtrainable_variables
Уregularization_losses
ћlayers
 ќlayer_regularization_losses
Ф	variables
§metrics
ўlayer_metrics
џnon_trainable_variables

o0
p1
 

o0
p1
Е
Цtrainable_variables
Чregularization_losses
layers
 layer_regularization_losses
Ш	variables
metrics
layer_metrics
non_trainable_variables

q0
r1
 

q0
r1
Е
Ъtrainable_variables
Ыregularization_losses
layers
 layer_regularization_losses
Ь	variables
metrics
layer_metrics
non_trainable_variables

s0
t1
 

s0
t1
Е
Юtrainable_variables
Яregularization_losses
layers
 layer_regularization_losses
а	variables
metrics
layer_metrics
non_trainable_variables

u0
v1
 

u0
v1
Е
вtrainable_variables
гregularization_losses
layers
 layer_regularization_losses
д	variables
metrics
layer_metrics
non_trainable_variables

w0
x1
 

w0
x1
Е
жtrainable_variables
зregularization_losses
layers
 layer_regularization_losses
и	variables
metrics
layer_metrics
non_trainable_variables

y0
z1
 

y0
z1
Е
кtrainable_variables
лregularization_losses
layers
 layer_regularization_losses
м	variables
metrics
layer_metrics
non_trainable_variables
 
 
 
Е
оtrainable_variables
пregularization_losses
layers
 layer_regularization_losses
р	variables
 metrics
Ёlayer_metrics
Ђnon_trainable_variables
 
 
 
Е
тtrainable_variables
уregularization_losses
Ѓlayers
 Єlayer_regularization_losses
ф	variables
Ѕmetrics
Іlayer_metrics
Їnon_trainable_variables
 
 
 
Е
цtrainable_variables
чregularization_losses
Јlayers
 Љlayer_regularization_losses
ш	variables
Њmetrics
Ћlayer_metrics
Ќnon_trainable_variables
 
 
 
Е
ъtrainable_variables
ыregularization_losses
­layers
 Ўlayer_regularization_losses
ь	variables
Џmetrics
Аlayer_metrics
Бnon_trainable_variables
 
 
 
Е
юtrainable_variables
яregularization_losses
Вlayers
 Гlayer_regularization_losses
№	variables
Дmetrics
Еlayer_metrics
Жnon_trainable_variables
 
 
 
Е
ђtrainable_variables
ѓregularization_losses
Зlayers
 Иlayer_regularization_losses
є	variables
Йmetrics
Кlayer_metrics
Лnon_trainable_variables
f
[0
\1
]2
^3
_4
`5
a6
b7
c8
d9
e10
f11
g12
h13
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
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

serving_default_input_2Placeholder*/
_output_shapes
:џџџџџџџџџ
*
dtype0*$
shape:џџџџџџџџџ


serving_default_input_3Placeholder*/
_output_shapes
:џџџџџџџџџ*
dtype0*$
shape:џџџџџџџџџ
Н
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3conv2d_2/kernelconv2d_2/biasconv2d/kernelconv2d/biasconv2d_3/kernelconv2d_3/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasencoder_output/kernelencoder_output/biasdense_23/kerneldense_23/biasdense_28/kerneldense_28/biasdense_26/kerneldense_26/biasdense_24/kerneldense_24/biasdense_29/kerneldense_29/biasdense_27/kerneldense_27/biasdense_25/kerneldense_25/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*@
_read_only_resource_inputs"
 	
 *0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_25054348
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
є

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp)encoder_output/kernel/Read/ReadVariableOp'encoder_output/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOpConst*+
Tin$
"2 *
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
GPU2*0J 8 **
f%R#
!__inference__traced_save_25055616

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_2/kernelconv2d_2/biasconv2d_1/kernelconv2d_1/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasencoder_output/kernelencoder_output/biasdense_23/kerneldense_23/biasdense_24/kerneldense_24/biasdense_26/kerneldense_26/biasdense_28/kerneldense_28/biasdense_25/kerneldense_25/biasdense_27/kerneldense_27/biasdense_29/kerneldense_29/bias**
Tin#
!2*
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
GPU2*0J 8 *-
f(R&
$__inference__traced_restore_25055716ья
Г
џ
-__inference_PASTA_CTAE_layer_call_fn_25054721
inputs_0
inputs_1
inputs_2!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:0 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18:

unknown_19: 

unknown_20:

unknown_21: 

unknown_22:

unknown_23:0

unknown_24:0

unknown_25:d

unknown_26:d

unknown_27:

unknown_28:
identity

identity_1

identity_2ЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*@
_read_only_resource_inputs"
 	
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_250536332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
inputs/1:YU
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
Г
џ
-__inference_PASTA_CTAE_layer_call_fn_25054792
inputs_0
inputs_1
inputs_2!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:0 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18:

unknown_19: 

unknown_20:

unknown_21: 

unknown_22:

unknown_23:0

unknown_24:0

unknown_25:d

unknown_26:d

unknown_27:

unknown_28:
identity

identity_1

identity_2ЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*@
_read_only_resource_inputs"
 	
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_250539692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
inputs/1:YU
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
е
H
,__inference_flatten_1_layer_call_fn_25054934

inputs
identityШ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250535122
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј

+__inference_dense_29_layer_call_fn_25055414

inputs
unknown:0
	unknown_0:0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_250529122
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ02

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х
a
E__inference_flatten_layer_call_and_return_conditional_losses_25054918

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25054857

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ч
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_25054929

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

c
G__inference_softmax_2_layer_call_and_return_conditional_losses_25055494

inputs
identity_
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:џџџџџџџџџ2	
Softmaxm
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_25053470

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ	*
ksize
*
paddingSAME*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
щ

K__inference_concatenate_2_layer_call_and_return_conditional_losses_25055002
inputs_0
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ02
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
э
L
0__inference_max_pooling2d_layer_call_fn_25054852

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_250534702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

§
D__inference_conv2d_layer_call_and_return_conditional_losses_25054803

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЛ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Ч
I
-__inference_reshape_10_layer_call_fn_25055431

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_250529962
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
=
у
E__inference_decoder_layer_call_and_return_conditional_losses_25053249

inputs#
dense_23_25053205:  
dense_23_25053207: #
dense_28_25053210: 
dense_28_25053212:#
dense_26_25053215: 
dense_26_25053217:#
dense_24_25053220: 
dense_24_25053222:#
dense_29_25053225:0
dense_29_25053227:0#
dense_27_25053230:d
dense_27_25053232:d#
dense_25_25053235:
dense_25_25053237:
identity

identity_1

identity_2Ђ dense_23/StatefulPartitionedCallЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ dense_27/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_25053205dense_23_25053207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_250528442"
 dense_23/StatefulPartitionedCallР
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_28_25053210dense_28_25053212*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_250528612"
 dense_28/StatefulPartitionedCallР
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_26_25053215dense_26_25053217*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_250528782"
 dense_26/StatefulPartitionedCallР
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_25053220dense_24_25053222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_250528952"
 dense_24/StatefulPartitionedCallР
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_25053225dense_29_25053227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_250529122"
 dense_29/StatefulPartitionedCallР
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_25053230dense_27_25053232*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_250529292"
 dense_27/StatefulPartitionedCallР
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_25053235dense_25_25053237*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_250529462"
 dense_25/StatefulPartitionedCall
reshape_12/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_12_layer_call_and_return_conditional_losses_250529662
reshape_12/PartitionedCall
reshape_11/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_250529822
reshape_11/PartitionedCall
reshape_10/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_250529962
reshape_10/PartitionedCall
softmax_2/PartitionedCallPartitionedCall#reshape_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_250530032
softmax_2/PartitionedCall
softmax_1/PartitionedCallPartitionedCall#reshape_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_250530102
softmax_1/PartitionedCallѓ
softmax/PartitionedCallPartitionedCall#reshape_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_250530172
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity"softmax_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity"softmax_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2У
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
n
ў

E__inference_decoder_layer_call_and_return_conditional_losses_25055114

inputs9
'dense_23_matmul_readvariableop_resource:  6
(dense_23_biasadd_readvariableop_resource: 9
'dense_28_matmul_readvariableop_resource: 6
(dense_28_biasadd_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource: 6
(dense_26_biasadd_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource:9
'dense_29_matmul_readvariableop_resource:06
(dense_29_biasadd_readvariableop_resource:09
'dense_27_matmul_readvariableop_resource:d6
(dense_27_biasadd_readvariableop_resource:d9
'dense_25_matmul_readvariableop_resource:6
(dense_25_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђdense_23/BiasAdd/ReadVariableOpЂdense_23/MatMul/ReadVariableOpЂdense_24/BiasAdd/ReadVariableOpЂdense_24/MatMul/ReadVariableOpЂdense_25/BiasAdd/ReadVariableOpЂdense_25/MatMul/ReadVariableOpЂdense_26/BiasAdd/ReadVariableOpЂdense_26/MatMul/ReadVariableOpЂdense_27/BiasAdd/ReadVariableOpЂdense_27/MatMul/ReadVariableOpЂdense_28/BiasAdd/ReadVariableOpЂdense_28/MatMul/ReadVariableOpЂdense_29/BiasAdd/ReadVariableOpЂdense_29/MatMul/ReadVariableOpЈ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_23/MatMul/ReadVariableOp
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_23/MatMulЇ
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_23/BiasAdd/ReadVariableOpЅ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_23/BiasAdds
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_23/ReluЈ
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_28/MatMul/ReadVariableOpЃ
dense_28/MatMulMatMuldense_23/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_28/MatMulЇ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_28/BiasAdd/ReadVariableOpЅ
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_28/ReluЈ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_26/MatMul/ReadVariableOpЃ
dense_26/MatMulMatMuldense_23/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_26/MatMulЇ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOpЅ
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_26/ReluЈ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOpЃ
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_24/MatMulЇ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpЅ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_24/BiasAdds
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_24/ReluЈ
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02 
dense_29/MatMul/ReadVariableOpЃ
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
dense_29/MatMulЇ
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_29/BiasAdd/ReadVariableOpЅ
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
dense_29/BiasAdds
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02
dense_29/ReluЈ
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_27/MatMul/ReadVariableOpЃ
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dense_27/MatMulЇ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_27/BiasAdd/ReadVariableOpЅ
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dense_27/ReluЈ
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_25/MatMul/ReadVariableOpЃ
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_25/MatMulЇ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOpЅ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_25/Reluo
reshape_12/ShapeShapedense_29/Relu:activations:0*
T0*
_output_shapes
:2
reshape_12/Shape
reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_12/strided_slice/stack
 reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_1
 reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_2Є
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
reshape_12/Reshape/shape/3ќ
reshape_12/Reshape/shapePack!reshape_12/strided_slice:output:0#reshape_12/Reshape/shape/1:output:0#reshape_12/Reshape/shape/2:output:0#reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_12/Reshape/shape­
reshape_12/ReshapeReshapedense_29/Relu:activations:0!reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
reshape_12/Reshapeo
reshape_11/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
reshape_11/Shape
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_11/strided_slice/stack
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_1
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_2Є
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
reshape_11/Reshape/shape/3ќ
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0#reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_11/Reshape/shape­
reshape_11/ReshapeReshapedense_27/Relu:activations:0!reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2
reshape_11/Reshapeo
reshape_10/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
reshape_10/Shape
reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_10/strided_slice/stack
 reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_1
 reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_2Є
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
reshape_10/Reshape/shape/1В
reshape_10/Reshape/shapePack!reshape_10/strided_slice:output:0#reshape_10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_10/Reshape/shapeЅ
reshape_10/ReshapeReshapedense_25/Relu:activations:0!reshape_10/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
reshape_10/Reshape
softmax_2/SoftmaxSoftmaxreshape_12/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
softmax_2/Softmax
softmax_1/SoftmaxSoftmaxreshape_11/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2
softmax_1/Softmax|
softmax/SoftmaxSoftmaxreshape_10/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
softmax/Softmaxt
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identitysoftmax_1/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identitysoftmax_2/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2Ѓ
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
n
ў

E__inference_decoder_layer_call_and_return_conditional_losses_25055200

inputs9
'dense_23_matmul_readvariableop_resource:  6
(dense_23_biasadd_readvariableop_resource: 9
'dense_28_matmul_readvariableop_resource: 6
(dense_28_biasadd_readvariableop_resource:9
'dense_26_matmul_readvariableop_resource: 6
(dense_26_biasadd_readvariableop_resource:9
'dense_24_matmul_readvariableop_resource: 6
(dense_24_biasadd_readvariableop_resource:9
'dense_29_matmul_readvariableop_resource:06
(dense_29_biasadd_readvariableop_resource:09
'dense_27_matmul_readvariableop_resource:d6
(dense_27_biasadd_readvariableop_resource:d9
'dense_25_matmul_readvariableop_resource:6
(dense_25_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђdense_23/BiasAdd/ReadVariableOpЂdense_23/MatMul/ReadVariableOpЂdense_24/BiasAdd/ReadVariableOpЂdense_24/MatMul/ReadVariableOpЂdense_25/BiasAdd/ReadVariableOpЂdense_25/MatMul/ReadVariableOpЂdense_26/BiasAdd/ReadVariableOpЂdense_26/MatMul/ReadVariableOpЂdense_27/BiasAdd/ReadVariableOpЂdense_27/MatMul/ReadVariableOpЂdense_28/BiasAdd/ReadVariableOpЂdense_28/MatMul/ReadVariableOpЂdense_29/BiasAdd/ReadVariableOpЂdense_29/MatMul/ReadVariableOpЈ
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense_23/MatMul/ReadVariableOp
dense_23/MatMulMatMulinputs&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_23/MatMulЇ
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_23/BiasAdd/ReadVariableOpЅ
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_23/BiasAdds
dense_23/ReluReludense_23/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
dense_23/ReluЈ
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_28/MatMul/ReadVariableOpЃ
dense_28/MatMulMatMuldense_23/Relu:activations:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_28/MatMulЇ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_28/BiasAdd/ReadVariableOpЅ
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_28/BiasAdds
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_28/ReluЈ
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_26/MatMul/ReadVariableOpЃ
dense_26/MatMulMatMuldense_23/Relu:activations:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_26/MatMulЇ
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_26/BiasAdd/ReadVariableOpЅ
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_26/BiasAdds
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_26/ReluЈ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_24/MatMul/ReadVariableOpЃ
dense_24/MatMulMatMuldense_23/Relu:activations:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_24/MatMulЇ
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpЅ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_24/BiasAdds
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_24/ReluЈ
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02 
dense_29/MatMul/ReadVariableOpЃ
dense_29/MatMulMatMuldense_28/Relu:activations:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
dense_29/MatMulЇ
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
dense_29/BiasAdd/ReadVariableOpЅ
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
dense_29/BiasAdds
dense_29/ReluReludense_29/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02
dense_29/ReluЈ
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_27/MatMul/ReadVariableOpЃ
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dense_27/MatMulЇ
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_27/BiasAdd/ReadVariableOpЅ
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dense_27/BiasAdds
dense_27/ReluReludense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
dense_27/ReluЈ
dense_25/MatMul/ReadVariableOpReadVariableOp'dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_25/MatMul/ReadVariableOpЃ
dense_25/MatMulMatMuldense_24/Relu:activations:0&dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_25/MatMulЇ
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOpЅ
dense_25/BiasAddBiasAdddense_25/MatMul:product:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_25/BiasAdds
dense_25/ReluReludense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_25/Reluo
reshape_12/ShapeShapedense_29/Relu:activations:0*
T0*
_output_shapes
:2
reshape_12/Shape
reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_12/strided_slice/stack
 reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_1
 reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_12/strided_slice/stack_2Є
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
reshape_12/Reshape/shape/3ќ
reshape_12/Reshape/shapePack!reshape_12/strided_slice:output:0#reshape_12/Reshape/shape/1:output:0#reshape_12/Reshape/shape/2:output:0#reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_12/Reshape/shape­
reshape_12/ReshapeReshapedense_29/Relu:activations:0!reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
reshape_12/Reshapeo
reshape_11/ShapeShapedense_27/Relu:activations:0*
T0*
_output_shapes
:2
reshape_11/Shape
reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_11/strided_slice/stack
 reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_1
 reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_11/strided_slice/stack_2Є
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
reshape_11/Reshape/shape/3ќ
reshape_11/Reshape/shapePack!reshape_11/strided_slice:output:0#reshape_11/Reshape/shape/1:output:0#reshape_11/Reshape/shape/2:output:0#reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_11/Reshape/shape­
reshape_11/ReshapeReshapedense_27/Relu:activations:0!reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2
reshape_11/Reshapeo
reshape_10/ShapeShapedense_25/Relu:activations:0*
T0*
_output_shapes
:2
reshape_10/Shape
reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
reshape_10/strided_slice/stack
 reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_1
 reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 reshape_10/strided_slice/stack_2Є
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
reshape_10/Reshape/shape/1В
reshape_10/Reshape/shapePack!reshape_10/strided_slice:output:0#reshape_10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_10/Reshape/shapeЅ
reshape_10/ReshapeReshapedense_25/Relu:activations:0!reshape_10/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
reshape_10/Reshape
softmax_2/SoftmaxSoftmaxreshape_12/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
softmax_2/Softmax
softmax_1/SoftmaxSoftmaxreshape_11/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2
softmax_1/Softmax|
softmax/SoftmaxSoftmaxreshape_10/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
softmax/Softmaxt
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identitysoftmax_1/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identitysoftmax_2/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2Ѓ
NoOpNoOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp ^dense_25/BiasAdd/ReadVariableOp^dense_25/MatMul/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ : : : : : : : : : : : : : : 2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp2B
dense_25/BiasAdd/ReadVariableOpdense_25/BiasAdd/ReadVariableOp2@
dense_25/MatMul/ReadVariableOpdense_25/MatMul/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

d
H__inference_reshape_11_layer_call_and_return_conditional_losses_25055445

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
strided_slice/stack_2т
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
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
х
a
E__inference_flatten_layer_call_and_return_conditional_losses_25053520

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
џ
*__inference_decoder_layer_call_fn_25053057
input_6
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:0
	unknown_8:0
	unknown_9:d

unknown_10:d

unknown_11:

unknown_12:
identity

identity_1

identity_2ЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_250530222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

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
/:џџџџџџџџџ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	input_6

c
G__inference_softmax_1_layer_call_and_return_conditional_losses_25055484

inputs
identity_
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:џџџџџџџџџ
2	
Softmaxm
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ
:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
ч
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_25053512

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
 
+__inference_conv2d_3_layer_call_fn_25054912

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250534832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

c
G__inference_softmax_2_layer_call_and_return_conditional_losses_25053003

inputs
identity_
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:џџџџџџџџџ2	
Softmaxm
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
M

H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25053969

inputs
inputs_1
inputs_2+
conv2d_2_25053890:
conv2d_2_25053892:)
conv2d_25053895:
conv2d_25053897:+
conv2d_3_25053902:
conv2d_3_25053904:+
conv2d_1_25053907:	
conv2d_1_25053909: 
dense_25053914:
dense_25053916:"
dense_1_25053919:
dense_1_25053921:"
dense_2_25053924:
dense_2_25053926:)
encoder_output_25053930:0 %
encoder_output_25053932: "
decoder_25053935:  
decoder_25053937: "
decoder_25053939: 
decoder_25053941:"
decoder_25053943: 
decoder_25053945:"
decoder_25053947: 
decoder_25053949:"
decoder_25053951:0
decoder_25053953:0"
decoder_25053955:d
decoder_25053957:d"
decoder_25053959:
decoder_25053961:
identity

identity_1

identity_2Ђconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdecoder/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ&encoder_output/StatefulPartitionedCallЇ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_2_25053890conv2d_2_25053892*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250534372"
 conv2d_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_25053895conv2d_25053897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_250534542 
conv2d/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_250534642!
max_pooling2d_1/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_250534702
max_pooling2d/PartitionedCallЧ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_25053902conv2d_3_25053904*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250534832"
 conv2d_3/StatefulPartitionedCallХ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_25053907conv2d_1_25053909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_250535002"
 conv2d_1/StatefulPartitionedCallџ
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250535122
flatten_1/PartitionedCallљ
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_250535202
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_25053914dense_25053916*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_250535332
dense/StatefulPartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_25053919dense_1_25053921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_250535502!
dense_1/StatefulPartitionedCallД
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_25053924dense_2_25053926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250535672!
dense_2/StatefulPartitionedCallо
concatenate_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_250535812
concatenate_2/PartitionedCallл
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0encoder_output_25053930encoder_output_25053932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_encoder_output_layer_call_and_return_conditional_losses_250535932(
&encoder_output/StatefulPartitionedCallщ
decoder/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0decoder_25053935decoder_25053937decoder_25053939decoder_25053941decoder_25053943decoder_25053945decoder_25053947decoder_25053949decoder_25053951decoder_25053953decoder_25053955decoder_25053957decoder_25053959decoder_25053961*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_250532492!
decoder/StatefulPartitionedCall
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity(decoder/StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity(decoder/StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^decoder/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ї
F__inference_dense_23_layer_call_and_return_conditional_losses_25055285

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

є
C__inference_dense_layer_call_and_return_conditional_losses_25053533

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
H__inference_reshape_12_layer_call_and_return_conditional_losses_25055464

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
strided_slice/stack_2т
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
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ0:O K
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs


1__inference_encoder_output_layer_call_fn_25055028

inputs
unknown:0 
	unknown_0: 
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_encoder_output_layer_call_and_return_conditional_losses_250535932
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ0: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_25054837

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І
ў
*__inference_decoder_layer_call_fn_25055237

inputs
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:0
	unknown_8:0
	unknown_9:d

unknown_10:d

unknown_11:

unknown_12:
identity

identity_1

identity_2ЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_250530222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

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
/:џџџџџџџџџ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

§
D__inference_conv2d_layer_call_and_return_conditional_losses_25053454

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЛ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
ё
N
2__inference_max_pooling2d_1_layer_call_fn_25054872

inputs
identityж
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_250534642
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
M

H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054275
input_1
input_2
input_3+
conv2d_2_25054196:
conv2d_2_25054198:)
conv2d_25054201:
conv2d_25054203:+
conv2d_3_25054208:
conv2d_3_25054210:+
conv2d_1_25054213:	
conv2d_1_25054215: 
dense_25054220:
dense_25054222:"
dense_1_25054225:
dense_1_25054227:"
dense_2_25054230:
dense_2_25054232:)
encoder_output_25054236:0 %
encoder_output_25054238: "
decoder_25054241:  
decoder_25054243: "
decoder_25054245: 
decoder_25054247:"
decoder_25054249: 
decoder_25054251:"
decoder_25054253: 
decoder_25054255:"
decoder_25054257:0
decoder_25054259:0"
decoder_25054261:d
decoder_25054263:d"
decoder_25054265:
decoder_25054267:
identity

identity_1

identity_2Ђconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdecoder/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ&encoder_output/StatefulPartitionedCallІ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_2_25054196conv2d_2_25054198*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250534372"
 conv2d_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_25054201conv2d_25054203*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_250534542 
conv2d/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_250534642!
max_pooling2d_1/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_250534702
max_pooling2d/PartitionedCallЧ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_25054208conv2d_3_25054210*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250534832"
 conv2d_3/StatefulPartitionedCallХ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_25054213conv2d_1_25054215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_250535002"
 conv2d_1/StatefulPartitionedCallџ
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250535122
flatten_1/PartitionedCallљ
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_250535202
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_25054220dense_25054222*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_250535332
dense/StatefulPartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_25054225dense_1_25054227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_250535502!
dense_1/StatefulPartitionedCallД
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_25054230dense_2_25054232*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250535672!
dense_2/StatefulPartitionedCallо
concatenate_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_250535812
concatenate_2/PartitionedCallл
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0encoder_output_25054236encoder_output_25054238*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_encoder_output_layer_call_and_return_conditional_losses_250535932(
&encoder_output/StatefulPartitionedCallщ
decoder/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0decoder_25054241decoder_25054243decoder_25054245decoder_25054247decoder_25054249decoder_25054251decoder_25054253decoder_25054255decoder_25054257decoder_25054259decoder_25054261decoder_25054263decoder_25054265decoder_25054267*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_250532492!
decoder/StatefulPartitionedCall
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity(decoder/StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity(decoder/StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^decoder/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:XT
/
_output_shapes
:џџџџџџџџџ

!
_user_specified_name	input_2:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3
M

H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25053633

inputs
inputs_1
inputs_2+
conv2d_2_25053438:
conv2d_2_25053440:)
conv2d_25053455:
conv2d_25053457:+
conv2d_3_25053484:
conv2d_3_25053486:+
conv2d_1_25053501:	
conv2d_1_25053503: 
dense_25053534:
dense_25053536:"
dense_1_25053551:
dense_1_25053553:"
dense_2_25053568:
dense_2_25053570:)
encoder_output_25053594:0 %
encoder_output_25053596: "
decoder_25053599:  
decoder_25053601: "
decoder_25053603: 
decoder_25053605:"
decoder_25053607: 
decoder_25053609:"
decoder_25053611: 
decoder_25053613:"
decoder_25053615:0
decoder_25053617:0"
decoder_25053619:d
decoder_25053621:d"
decoder_25053623:
decoder_25053625:
identity

identity_1

identity_2Ђconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdecoder/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ&encoder_output/StatefulPartitionedCallЇ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2conv2d_2_25053438conv2d_2_25053440*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250534372"
 conv2d_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs_1conv2d_25053455conv2d_25053457*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_250534542 
conv2d/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_250534642!
max_pooling2d_1/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_250534702
max_pooling2d/PartitionedCallЧ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_25053484conv2d_3_25053486*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250534832"
 conv2d_3/StatefulPartitionedCallХ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_25053501conv2d_1_25053503*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_250535002"
 conv2d_1/StatefulPartitionedCallџ
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250535122
flatten_1/PartitionedCallљ
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_250535202
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_25053534dense_25053536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_250535332
dense/StatefulPartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_25053551dense_1_25053553*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_250535502!
dense_1/StatefulPartitionedCallД
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_25053568dense_2_25053570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250535672!
dense_2/StatefulPartitionedCallо
concatenate_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_250535812
concatenate_2/PartitionedCallл
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0encoder_output_25053594encoder_output_25053596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_encoder_output_layer_call_and_return_conditional_losses_250535932(
&encoder_output/StatefulPartitionedCallщ
decoder/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0decoder_25053599decoder_25053601decoder_25053603decoder_25053605decoder_25053607decoder_25053609decoder_25053611decoder_25053613decoder_25053615decoder_25053617decoder_25053619decoder_25053621decoder_25053623decoder_25053625*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_250530222!
decoder/StatefulPartitionedCall
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity(decoder/StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity(decoder/StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^decoder/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:WS
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў
ѕ
&__inference_signature_wrapper_25054348
input_1
input_2
input_3!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:0 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18:

unknown_19: 

unknown_20:

unknown_21: 

unknown_22:

unknown_23:0

unknown_24:0

unknown_25:d

unknown_26:d

unknown_27:

unknown_28:
identity

identity_1

identity_2ЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*@
_read_only_resource_inputs"
 	
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_250527822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:XT
/
_output_shapes
:џџџџџџџџџ

!
_user_specified_name	input_2:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3

c
G__inference_softmax_1_layer_call_and_return_conditional_losses_25053010

inputs
identity_
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:џџџџџџџџџ
2	
Softmaxm
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ
:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
ј

+__inference_dense_25_layer_call_fn_25055374

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_250529462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э
џ
F__inference_conv2d_3_layer_call_and_return_conditional_losses_25054903

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25053464

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ёз
Ы
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054499
inputs_0
inputs_1
inputs_2A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:	6
(conv2d_1_biasadd_readvariableop_resource:6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:?
-encoder_output_matmul_readvariableop_resource:0 <
.encoder_output_biasadd_readvariableop_resource: A
/decoder_dense_23_matmul_readvariableop_resource:  >
0decoder_dense_23_biasadd_readvariableop_resource: A
/decoder_dense_28_matmul_readvariableop_resource: >
0decoder_dense_28_biasadd_readvariableop_resource:A
/decoder_dense_26_matmul_readvariableop_resource: >
0decoder_dense_26_biasadd_readvariableop_resource:A
/decoder_dense_24_matmul_readvariableop_resource: >
0decoder_dense_24_biasadd_readvariableop_resource:A
/decoder_dense_29_matmul_readvariableop_resource:0>
0decoder_dense_29_biasadd_readvariableop_resource:0A
/decoder_dense_27_matmul_readvariableop_resource:d>
0decoder_dense_27_biasadd_readvariableop_resource:dA
/decoder_dense_25_matmul_readvariableop_resource:>
0decoder_dense_25_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂ'decoder/dense_23/BiasAdd/ReadVariableOpЂ&decoder/dense_23/MatMul/ReadVariableOpЂ'decoder/dense_24/BiasAdd/ReadVariableOpЂ&decoder/dense_24/MatMul/ReadVariableOpЂ'decoder/dense_25/BiasAdd/ReadVariableOpЂ&decoder/dense_25/MatMul/ReadVariableOpЂ'decoder/dense_26/BiasAdd/ReadVariableOpЂ&decoder/dense_26/MatMul/ReadVariableOpЂ'decoder/dense_27/BiasAdd/ReadVariableOpЂ&decoder/dense_27/MatMul/ReadVariableOpЂ'decoder/dense_28/BiasAdd/ReadVariableOpЂ&decoder/dense_28/MatMul/ReadVariableOpЂ'decoder/dense_29/BiasAdd/ReadVariableOpЂ&decoder/dense_29/MatMul/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂ%encoder_output/BiasAdd/ReadVariableOpЂ$encoder_output/MatMul/ReadVariableOpА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpи
conv2d_2/Conv2DConv2Dinputs_2&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpУ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_2/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpв
conv2d/Conv2DConv2Dinputs_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЛ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
conv2d/ReluЦ
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPoolР
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ	*
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPoolА
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpй
conv2d_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2d_3/Conv2DЇ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpЌ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_3/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_1/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_1/Const
flatten_1/ReshapeReshapeconv2d_3/Relu:activations:0flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten/Const
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
flatten/Reshape
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

dense/ReluЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/ReluЅ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/Relux
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisщ
concatenate_2/concatConcatV2dense/Relu:activations:0dense_1/Relu:activations:0dense_2/Relu:activations:0"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ02
concatenate_2/concatК
$encoder_output/MatMul/ReadVariableOpReadVariableOp-encoder_output_matmul_readvariableop_resource*
_output_shapes

:0 *
dtype02&
$encoder_output/MatMul/ReadVariableOpЗ
encoder_output/MatMulMatMulconcatenate_2/concat:output:0,encoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder_output/MatMulЙ
%encoder_output/BiasAdd/ReadVariableOpReadVariableOp.encoder_output_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%encoder_output/BiasAdd/ReadVariableOpН
encoder_output/BiasAddBiasAddencoder_output/MatMul:product:0-encoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder_output/BiasAddР
&decoder/dense_23/MatMul/ReadVariableOpReadVariableOp/decoder_dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02(
&decoder/dense_23/MatMul/ReadVariableOpП
decoder/dense_23/MatMulMatMulencoder_output/BiasAdd:output:0.decoder/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
decoder/dense_23/MatMulП
'decoder/dense_23/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'decoder/dense_23/BiasAdd/ReadVariableOpХ
decoder/dense_23/BiasAddBiasAdd!decoder/dense_23/MatMul:product:0/decoder/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
decoder/dense_23/BiasAdd
decoder/dense_23/ReluRelu!decoder/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
decoder/dense_23/ReluР
&decoder/dense_28/MatMul/ReadVariableOpReadVariableOp/decoder_dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_28/MatMul/ReadVariableOpУ
decoder/dense_28/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_28/MatMulП
'decoder/dense_28/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_28/BiasAdd/ReadVariableOpХ
decoder/dense_28/BiasAddBiasAdd!decoder/dense_28/MatMul:product:0/decoder/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_28/BiasAdd
decoder/dense_28/ReluRelu!decoder/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_28/ReluР
&decoder/dense_26/MatMul/ReadVariableOpReadVariableOp/decoder_dense_26_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_26/MatMul/ReadVariableOpУ
decoder/dense_26/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_26/MatMulП
'decoder/dense_26/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_26/BiasAdd/ReadVariableOpХ
decoder/dense_26/BiasAddBiasAdd!decoder/dense_26/MatMul:product:0/decoder/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_26/BiasAdd
decoder/dense_26/ReluRelu!decoder/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_26/ReluР
&decoder/dense_24/MatMul/ReadVariableOpReadVariableOp/decoder_dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_24/MatMul/ReadVariableOpУ
decoder/dense_24/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_24/MatMulП
'decoder/dense_24/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_24/BiasAdd/ReadVariableOpХ
decoder/dense_24/BiasAddBiasAdd!decoder/dense_24/MatMul:product:0/decoder/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_24/BiasAdd
decoder/dense_24/ReluRelu!decoder/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_24/ReluР
&decoder/dense_29/MatMul/ReadVariableOpReadVariableOp/decoder_dense_29_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02(
&decoder/dense_29/MatMul/ReadVariableOpУ
decoder/dense_29/MatMulMatMul#decoder/dense_28/Relu:activations:0.decoder/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
decoder/dense_29/MatMulП
'decoder/dense_29/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_29_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'decoder/dense_29/BiasAdd/ReadVariableOpХ
decoder/dense_29/BiasAddBiasAdd!decoder/dense_29/MatMul:product:0/decoder/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
decoder/dense_29/BiasAdd
decoder/dense_29/ReluRelu!decoder/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02
decoder/dense_29/ReluР
&decoder/dense_27/MatMul/ReadVariableOpReadVariableOp/decoder_dense_27_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02(
&decoder/dense_27/MatMul/ReadVariableOpУ
decoder/dense_27/MatMulMatMul#decoder/dense_26/Relu:activations:0.decoder/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
decoder/dense_27/MatMulП
'decoder/dense_27/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_27_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'decoder/dense_27/BiasAdd/ReadVariableOpХ
decoder/dense_27/BiasAddBiasAdd!decoder/dense_27/MatMul:product:0/decoder/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
decoder/dense_27/BiasAdd
decoder/dense_27/ReluRelu!decoder/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
decoder/dense_27/ReluР
&decoder/dense_25/MatMul/ReadVariableOpReadVariableOp/decoder_dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&decoder/dense_25/MatMul/ReadVariableOpУ
decoder/dense_25/MatMulMatMul#decoder/dense_24/Relu:activations:0.decoder/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_25/MatMulП
'decoder/dense_25/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_25/BiasAdd/ReadVariableOpХ
decoder/dense_25/BiasAddBiasAdd!decoder/dense_25/MatMul:product:0/decoder/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_25/BiasAdd
decoder/dense_25/ReluRelu!decoder/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_25/Relu
decoder/reshape_12/ShapeShape#decoder/dense_29/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_12/Shape
&decoder/reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_12/strided_slice/stack
(decoder/reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_12/strided_slice/stack_1
(decoder/reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_12/strided_slice/stack_2д
 decoder/reshape_12/strided_sliceStridedSlice!decoder/reshape_12/Shape:output:0/decoder/reshape_12/strided_slice/stack:output:01decoder/reshape_12/strided_slice/stack_1:output:01decoder/reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_12/strided_slice
"decoder/reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/1
"decoder/reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/2
"decoder/reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/3Ќ
 decoder/reshape_12/Reshape/shapePack)decoder/reshape_12/strided_slice:output:0+decoder/reshape_12/Reshape/shape/1:output:0+decoder/reshape_12/Reshape/shape/2:output:0+decoder/reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_12/Reshape/shapeЭ
decoder/reshape_12/ReshapeReshape#decoder/dense_29/Relu:activations:0)decoder/reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
decoder/reshape_12/Reshape
decoder/reshape_11/ShapeShape#decoder/dense_27/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_11/Shape
&decoder/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_11/strided_slice/stack
(decoder/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_11/strided_slice/stack_1
(decoder/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_11/strided_slice/stack_2д
 decoder/reshape_11/strided_sliceStridedSlice!decoder/reshape_11/Shape:output:0/decoder/reshape_11/strided_slice/stack:output:01decoder/reshape_11/strided_slice/stack_1:output:01decoder/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_11/strided_slice
"decoder/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_11/Reshape/shape/1
"decoder/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_11/Reshape/shape/2
"decoder/reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
2$
"decoder/reshape_11/Reshape/shape/3Ќ
 decoder/reshape_11/Reshape/shapePack)decoder/reshape_11/strided_slice:output:0+decoder/reshape_11/Reshape/shape/1:output:0+decoder/reshape_11/Reshape/shape/2:output:0+decoder/reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_11/Reshape/shapeЭ
decoder/reshape_11/ReshapeReshape#decoder/dense_27/Relu:activations:0)decoder/reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2
decoder/reshape_11/Reshape
decoder/reshape_10/ShapeShape#decoder/dense_25/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_10/Shape
&decoder/reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_10/strided_slice/stack
(decoder/reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_10/strided_slice/stack_1
(decoder/reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_10/strided_slice/stack_2д
 decoder/reshape_10/strided_sliceStridedSlice!decoder/reshape_10/Shape:output:0/decoder/reshape_10/strided_slice/stack:output:01decoder/reshape_10/strided_slice/stack_1:output:01decoder/reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_10/strided_slice
"decoder/reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_10/Reshape/shape/1в
 decoder/reshape_10/Reshape/shapePack)decoder/reshape_10/strided_slice:output:0+decoder/reshape_10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_10/Reshape/shapeХ
decoder/reshape_10/ReshapeReshape#decoder/dense_25/Relu:activations:0)decoder/reshape_10/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/reshape_10/Reshape 
decoder/softmax_2/SoftmaxSoftmax#decoder/reshape_12/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
decoder/softmax_2/Softmax 
decoder/softmax_1/SoftmaxSoftmax#decoder/reshape_11/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2
decoder/softmax_1/Softmax
decoder/softmax/SoftmaxSoftmax#decoder/reshape_10/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/softmax/Softmax|
IdentityIdentity!decoder/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity#decoder/softmax_1/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity#decoder/softmax_2/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2Љ	
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^decoder/dense_23/BiasAdd/ReadVariableOp'^decoder/dense_23/MatMul/ReadVariableOp(^decoder/dense_24/BiasAdd/ReadVariableOp'^decoder/dense_24/MatMul/ReadVariableOp(^decoder/dense_25/BiasAdd/ReadVariableOp'^decoder/dense_25/MatMul/ReadVariableOp(^decoder/dense_26/BiasAdd/ReadVariableOp'^decoder/dense_26/MatMul/ReadVariableOp(^decoder/dense_27/BiasAdd/ReadVariableOp'^decoder/dense_27/MatMul/ReadVariableOp(^decoder/dense_28/BiasAdd/ReadVariableOp'^decoder/dense_28/MatMul/ReadVariableOp(^decoder/dense_29/BiasAdd/ReadVariableOp'^decoder/dense_29/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^encoder_output/BiasAdd/ReadVariableOp%^encoder_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2R
'decoder/dense_23/BiasAdd/ReadVariableOp'decoder/dense_23/BiasAdd/ReadVariableOp2P
&decoder/dense_23/MatMul/ReadVariableOp&decoder/dense_23/MatMul/ReadVariableOp2R
'decoder/dense_24/BiasAdd/ReadVariableOp'decoder/dense_24/BiasAdd/ReadVariableOp2P
&decoder/dense_24/MatMul/ReadVariableOp&decoder/dense_24/MatMul/ReadVariableOp2R
'decoder/dense_25/BiasAdd/ReadVariableOp'decoder/dense_25/BiasAdd/ReadVariableOp2P
&decoder/dense_25/MatMul/ReadVariableOp&decoder/dense_25/MatMul/ReadVariableOp2R
'decoder/dense_26/BiasAdd/ReadVariableOp'decoder/dense_26/BiasAdd/ReadVariableOp2P
&decoder/dense_26/MatMul/ReadVariableOp&decoder/dense_26/MatMul/ReadVariableOp2R
'decoder/dense_27/BiasAdd/ReadVariableOp'decoder/dense_27/BiasAdd/ReadVariableOp2P
&decoder/dense_27/MatMul/ReadVariableOp&decoder/dense_27/MatMul/ReadVariableOp2R
'decoder/dense_28/BiasAdd/ReadVariableOp'decoder/dense_28/BiasAdd/ReadVariableOp2P
&decoder/dense_28/MatMul/ReadVariableOp&decoder/dense_28/MatMul/ReadVariableOp2R
'decoder/dense_29/BiasAdd/ReadVariableOp'decoder/dense_29/BiasAdd/ReadVariableOp2P
&decoder/dense_29/MatMul/ReadVariableOp&decoder/dense_29/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2N
%encoder_output/BiasAdd/ReadVariableOp%encoder_output/BiasAdd/ReadVariableOp2L
$encoder_output/MatMul/ReadVariableOp$encoder_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
inputs/1:YU
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
з
I
-__inference_reshape_11_layer_call_fn_25055450

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_250529822
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs

ї
F__inference_dense_27_layer_call_and_return_conditional_losses_25055385

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

џ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_25053437

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЛ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

і
E__inference_dense_2_layer_call_and_return_conditional_losses_25053567

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ёз
Ы
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054650
inputs_0
inputs_1
inputs_2A
'conv2d_2_conv2d_readvariableop_resource:6
(conv2d_2_biasadd_readvariableop_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:	6
(conv2d_1_biasadd_readvariableop_resource:6
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:?
-encoder_output_matmul_readvariableop_resource:0 <
.encoder_output_biasadd_readvariableop_resource: A
/decoder_dense_23_matmul_readvariableop_resource:  >
0decoder_dense_23_biasadd_readvariableop_resource: A
/decoder_dense_28_matmul_readvariableop_resource: >
0decoder_dense_28_biasadd_readvariableop_resource:A
/decoder_dense_26_matmul_readvariableop_resource: >
0decoder_dense_26_biasadd_readvariableop_resource:A
/decoder_dense_24_matmul_readvariableop_resource: >
0decoder_dense_24_biasadd_readvariableop_resource:A
/decoder_dense_29_matmul_readvariableop_resource:0>
0decoder_dense_29_biasadd_readvariableop_resource:0A
/decoder_dense_27_matmul_readvariableop_resource:d>
0decoder_dense_27_biasadd_readvariableop_resource:dA
/decoder_dense_25_matmul_readvariableop_resource:>
0decoder_dense_25_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђconv2d/BiasAdd/ReadVariableOpЂconv2d/Conv2D/ReadVariableOpЂconv2d_1/BiasAdd/ReadVariableOpЂconv2d_1/Conv2D/ReadVariableOpЂconv2d_2/BiasAdd/ReadVariableOpЂconv2d_2/Conv2D/ReadVariableOpЂconv2d_3/BiasAdd/ReadVariableOpЂconv2d_3/Conv2D/ReadVariableOpЂ'decoder/dense_23/BiasAdd/ReadVariableOpЂ&decoder/dense_23/MatMul/ReadVariableOpЂ'decoder/dense_24/BiasAdd/ReadVariableOpЂ&decoder/dense_24/MatMul/ReadVariableOpЂ'decoder/dense_25/BiasAdd/ReadVariableOpЂ&decoder/dense_25/MatMul/ReadVariableOpЂ'decoder/dense_26/BiasAdd/ReadVariableOpЂ&decoder/dense_26/MatMul/ReadVariableOpЂ'decoder/dense_27/BiasAdd/ReadVariableOpЂ&decoder/dense_27/MatMul/ReadVariableOpЂ'decoder/dense_28/BiasAdd/ReadVariableOpЂ&decoder/dense_28/MatMul/ReadVariableOpЂ'decoder/dense_29/BiasAdd/ReadVariableOpЂ&decoder/dense_29/MatMul/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂdense_2/BiasAdd/ReadVariableOpЂdense_2/MatMul/ReadVariableOpЂ%encoder_output/BiasAdd/ReadVariableOpЂ$encoder_output/MatMul/ReadVariableOpА
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_2/Conv2D/ReadVariableOpи
conv2d_2/Conv2DConv2Dinputs_2&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
2
conv2d_2/Conv2DЇ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpУ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_2/ReluЊ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpв
conv2d/Conv2DConv2Dinputs_1$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides
2
conv2d/Conv2DЁ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOpЛ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
conv2d/ReluЦ
max_pooling2d_1/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPoolР
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ	*
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPoolА
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_3/Conv2D/ReadVariableOpй
conv2d_3/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2d_3/Conv2DЇ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpЌ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_3/ReluА
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02 
conv2d_1/Conv2D/ReadVariableOpз
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
conv2d_1/Conv2DЇ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpЌ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2d_1/Relus
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_1/Const
flatten_1/ReshapeReshapeconv2d_3/Relu:activations:0flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten/Const
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
flatten/Reshape
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

dense/ReluЅ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/MatMulЄ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpЁ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_1/ReluЅ
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/MatMulЄ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpЁ
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_2/Relux
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisщ
concatenate_2/concatConcatV2dense/Relu:activations:0dense_1/Relu:activations:0dense_2/Relu:activations:0"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ02
concatenate_2/concatК
$encoder_output/MatMul/ReadVariableOpReadVariableOp-encoder_output_matmul_readvariableop_resource*
_output_shapes

:0 *
dtype02&
$encoder_output/MatMul/ReadVariableOpЗ
encoder_output/MatMulMatMulconcatenate_2/concat:output:0,encoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder_output/MatMulЙ
%encoder_output/BiasAdd/ReadVariableOpReadVariableOp.encoder_output_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%encoder_output/BiasAdd/ReadVariableOpН
encoder_output/BiasAddBiasAddencoder_output/MatMul:product:0-encoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
encoder_output/BiasAddР
&decoder/dense_23/MatMul/ReadVariableOpReadVariableOp/decoder_dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02(
&decoder/dense_23/MatMul/ReadVariableOpП
decoder/dense_23/MatMulMatMulencoder_output/BiasAdd:output:0.decoder/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
decoder/dense_23/MatMulП
'decoder/dense_23/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'decoder/dense_23/BiasAdd/ReadVariableOpХ
decoder/dense_23/BiasAddBiasAdd!decoder/dense_23/MatMul:product:0/decoder/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
decoder/dense_23/BiasAdd
decoder/dense_23/ReluRelu!decoder/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
decoder/dense_23/ReluР
&decoder/dense_28/MatMul/ReadVariableOpReadVariableOp/decoder_dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_28/MatMul/ReadVariableOpУ
decoder/dense_28/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_28/MatMulП
'decoder/dense_28/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_28/BiasAdd/ReadVariableOpХ
decoder/dense_28/BiasAddBiasAdd!decoder/dense_28/MatMul:product:0/decoder/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_28/BiasAdd
decoder/dense_28/ReluRelu!decoder/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_28/ReluР
&decoder/dense_26/MatMul/ReadVariableOpReadVariableOp/decoder_dense_26_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_26/MatMul/ReadVariableOpУ
decoder/dense_26/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_26/MatMulП
'decoder/dense_26/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_26/BiasAdd/ReadVariableOpХ
decoder/dense_26/BiasAddBiasAdd!decoder/dense_26/MatMul:product:0/decoder/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_26/BiasAdd
decoder/dense_26/ReluRelu!decoder/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_26/ReluР
&decoder/dense_24/MatMul/ReadVariableOpReadVariableOp/decoder_dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype02(
&decoder/dense_24/MatMul/ReadVariableOpУ
decoder/dense_24/MatMulMatMul#decoder/dense_23/Relu:activations:0.decoder/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_24/MatMulП
'decoder/dense_24/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_24/BiasAdd/ReadVariableOpХ
decoder/dense_24/BiasAddBiasAdd!decoder/dense_24/MatMul:product:0/decoder/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_24/BiasAdd
decoder/dense_24/ReluRelu!decoder/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_24/ReluР
&decoder/dense_29/MatMul/ReadVariableOpReadVariableOp/decoder_dense_29_matmul_readvariableop_resource*
_output_shapes

:0*
dtype02(
&decoder/dense_29/MatMul/ReadVariableOpУ
decoder/dense_29/MatMulMatMul#decoder/dense_28/Relu:activations:0.decoder/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
decoder/dense_29/MatMulП
'decoder/dense_29/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_29_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02)
'decoder/dense_29/BiasAdd/ReadVariableOpХ
decoder/dense_29/BiasAddBiasAdd!decoder/dense_29/MatMul:product:0/decoder/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
decoder/dense_29/BiasAdd
decoder/dense_29/ReluRelu!decoder/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02
decoder/dense_29/ReluР
&decoder/dense_27/MatMul/ReadVariableOpReadVariableOp/decoder_dense_27_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02(
&decoder/dense_27/MatMul/ReadVariableOpУ
decoder/dense_27/MatMulMatMul#decoder/dense_26/Relu:activations:0.decoder/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
decoder/dense_27/MatMulП
'decoder/dense_27/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_27_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'decoder/dense_27/BiasAdd/ReadVariableOpХ
decoder/dense_27/BiasAddBiasAdd!decoder/dense_27/MatMul:product:0/decoder/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
decoder/dense_27/BiasAdd
decoder/dense_27/ReluRelu!decoder/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
decoder/dense_27/ReluР
&decoder/dense_25/MatMul/ReadVariableOpReadVariableOp/decoder_dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&decoder/dense_25/MatMul/ReadVariableOpУ
decoder/dense_25/MatMulMatMul#decoder/dense_24/Relu:activations:0.decoder/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_25/MatMulП
'decoder/dense_25/BiasAdd/ReadVariableOpReadVariableOp0decoder_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/dense_25/BiasAdd/ReadVariableOpХ
decoder/dense_25/BiasAddBiasAdd!decoder/dense_25/MatMul:product:0/decoder/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_25/BiasAdd
decoder/dense_25/ReluRelu!decoder/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/dense_25/Relu
decoder/reshape_12/ShapeShape#decoder/dense_29/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_12/Shape
&decoder/reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_12/strided_slice/stack
(decoder/reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_12/strided_slice/stack_1
(decoder/reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_12/strided_slice/stack_2д
 decoder/reshape_12/strided_sliceStridedSlice!decoder/reshape_12/Shape:output:0/decoder/reshape_12/strided_slice/stack:output:01decoder/reshape_12/strided_slice/stack_1:output:01decoder/reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_12/strided_slice
"decoder/reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/1
"decoder/reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/2
"decoder/reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_12/Reshape/shape/3Ќ
 decoder/reshape_12/Reshape/shapePack)decoder/reshape_12/strided_slice:output:0+decoder/reshape_12/Reshape/shape/1:output:0+decoder/reshape_12/Reshape/shape/2:output:0+decoder/reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_12/Reshape/shapeЭ
decoder/reshape_12/ReshapeReshape#decoder/dense_29/Relu:activations:0)decoder/reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
decoder/reshape_12/Reshape
decoder/reshape_11/ShapeShape#decoder/dense_27/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_11/Shape
&decoder/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_11/strided_slice/stack
(decoder/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_11/strided_slice/stack_1
(decoder/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_11/strided_slice/stack_2д
 decoder/reshape_11/strided_sliceStridedSlice!decoder/reshape_11/Shape:output:0/decoder/reshape_11/strided_slice/stack:output:01decoder/reshape_11/strided_slice/stack_1:output:01decoder/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_11/strided_slice
"decoder/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_11/Reshape/shape/1
"decoder/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_11/Reshape/shape/2
"decoder/reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
2$
"decoder/reshape_11/Reshape/shape/3Ќ
 decoder/reshape_11/Reshape/shapePack)decoder/reshape_11/strided_slice:output:0+decoder/reshape_11/Reshape/shape/1:output:0+decoder/reshape_11/Reshape/shape/2:output:0+decoder/reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_11/Reshape/shapeЭ
decoder/reshape_11/ReshapeReshape#decoder/dense_27/Relu:activations:0)decoder/reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2
decoder/reshape_11/Reshape
decoder/reshape_10/ShapeShape#decoder/dense_25/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape_10/Shape
&decoder/reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&decoder/reshape_10/strided_slice/stack
(decoder/reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_10/strided_slice/stack_1
(decoder/reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(decoder/reshape_10/strided_slice/stack_2д
 decoder/reshape_10/strided_sliceStridedSlice!decoder/reshape_10/Shape:output:0/decoder/reshape_10/strided_slice/stack:output:01decoder/reshape_10/strided_slice/stack_1:output:01decoder/reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 decoder/reshape_10/strided_slice
"decoder/reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/reshape_10/Reshape/shape/1в
 decoder/reshape_10/Reshape/shapePack)decoder/reshape_10/strided_slice:output:0+decoder/reshape_10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2"
 decoder/reshape_10/Reshape/shapeХ
decoder/reshape_10/ReshapeReshape#decoder/dense_25/Relu:activations:0)decoder/reshape_10/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/reshape_10/Reshape 
decoder/softmax_2/SoftmaxSoftmax#decoder/reshape_12/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
decoder/softmax_2/Softmax 
decoder/softmax_1/SoftmaxSoftmax#decoder/reshape_11/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2
decoder/softmax_1/Softmax
decoder/softmax/SoftmaxSoftmax#decoder/reshape_10/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
decoder/softmax/Softmax|
IdentityIdentity!decoder/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity#decoder/softmax_1/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity#decoder/softmax_2/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2Љ	
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp(^decoder/dense_23/BiasAdd/ReadVariableOp'^decoder/dense_23/MatMul/ReadVariableOp(^decoder/dense_24/BiasAdd/ReadVariableOp'^decoder/dense_24/MatMul/ReadVariableOp(^decoder/dense_25/BiasAdd/ReadVariableOp'^decoder/dense_25/MatMul/ReadVariableOp(^decoder/dense_26/BiasAdd/ReadVariableOp'^decoder/dense_26/MatMul/ReadVariableOp(^decoder/dense_27/BiasAdd/ReadVariableOp'^decoder/dense_27/MatMul/ReadVariableOp(^decoder/dense_28/BiasAdd/ReadVariableOp'^decoder/dense_28/MatMul/ReadVariableOp(^decoder/dense_29/BiasAdd/ReadVariableOp'^decoder/dense_29/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^encoder_output/BiasAdd/ReadVariableOp%^encoder_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2R
'decoder/dense_23/BiasAdd/ReadVariableOp'decoder/dense_23/BiasAdd/ReadVariableOp2P
&decoder/dense_23/MatMul/ReadVariableOp&decoder/dense_23/MatMul/ReadVariableOp2R
'decoder/dense_24/BiasAdd/ReadVariableOp'decoder/dense_24/BiasAdd/ReadVariableOp2P
&decoder/dense_24/MatMul/ReadVariableOp&decoder/dense_24/MatMul/ReadVariableOp2R
'decoder/dense_25/BiasAdd/ReadVariableOp'decoder/dense_25/BiasAdd/ReadVariableOp2P
&decoder/dense_25/MatMul/ReadVariableOp&decoder/dense_25/MatMul/ReadVariableOp2R
'decoder/dense_26/BiasAdd/ReadVariableOp'decoder/dense_26/BiasAdd/ReadVariableOp2P
&decoder/dense_26/MatMul/ReadVariableOp&decoder/dense_26/MatMul/ReadVariableOp2R
'decoder/dense_27/BiasAdd/ReadVariableOp'decoder/dense_27/BiasAdd/ReadVariableOp2P
&decoder/dense_27/MatMul/ReadVariableOp&decoder/dense_27/MatMul/ReadVariableOp2R
'decoder/dense_28/BiasAdd/ReadVariableOp'decoder/dense_28/BiasAdd/ReadVariableOp2P
&decoder/dense_28/MatMul/ReadVariableOp&decoder/dense_28/MatMul/ReadVariableOp2R
'decoder/dense_29/BiasAdd/ReadVariableOp'decoder/dense_29/BiasAdd/ReadVariableOp2P
&decoder/dense_29/MatMul/ReadVariableOp&decoder/dense_29/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2N
%encoder_output/BiasAdd/ReadVariableOp%encoder_output/BiasAdd/ReadVariableOp2L
$encoder_output/MatMul/ReadVariableOp$encoder_output/MatMul/ReadVariableOp:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
inputs/1:YU
/
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
П
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25054862

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё=
ф
E__inference_decoder_layer_call_and_return_conditional_losses_25053368
input_6#
dense_23_25053324:  
dense_23_25053326: #
dense_28_25053329: 
dense_28_25053331:#
dense_26_25053334: 
dense_26_25053336:#
dense_24_25053339: 
dense_24_25053341:#
dense_29_25053344:0
dense_29_25053346:0#
dense_27_25053349:d
dense_27_25053351:d#
dense_25_25053354:
dense_25_25053356:
identity

identity_1

identity_2Ђ dense_23/StatefulPartitionedCallЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ dense_27/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_23_25053324dense_23_25053326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_250528442"
 dense_23/StatefulPartitionedCallР
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_28_25053329dense_28_25053331*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_250528612"
 dense_28/StatefulPartitionedCallР
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_26_25053334dense_26_25053336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_250528782"
 dense_26/StatefulPartitionedCallР
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_25053339dense_24_25053341*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_250528952"
 dense_24/StatefulPartitionedCallР
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_25053344dense_29_25053346*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_250529122"
 dense_29/StatefulPartitionedCallР
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_25053349dense_27_25053351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_250529292"
 dense_27/StatefulPartitionedCallР
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_25053354dense_25_25053356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_250529462"
 dense_25/StatefulPartitionedCall
reshape_12/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_12_layer_call_and_return_conditional_losses_250529662
reshape_12/PartitionedCall
reshape_11/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_250529822
reshape_11/PartitionedCall
reshape_10/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_250529962
reshape_10/PartitionedCall
softmax_2/PartitionedCallPartitionedCall#reshape_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_250530032
softmax_2/PartitionedCall
softmax_1/PartitionedCallPartitionedCall#reshape_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_250530102
softmax_1/PartitionedCallѓ
softmax/PartitionedCallPartitionedCall#reshape_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_250530172
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity"softmax_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity"softmax_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2У
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	input_6
п

K__inference_concatenate_2_layer_call_and_return_conditional_losses_25053581

inputs
inputs_1
inputs_2
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ02
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х
H
,__inference_softmax_2_layer_call_fn_25055499

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_250530032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
э
џ
F__inference_conv2d_3_layer_call_and_return_conditional_losses_25053483

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ
џ
*__inference_decoder_layer_call_fn_25053321
input_6
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:0
	unknown_8:0
	unknown_9:d

unknown_10:d

unknown_11:

unknown_12:
identity

identity_1

identity_2ЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_250532492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

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
/:џџџџџџџџџ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	input_6
б
j
0__inference_concatenate_2_layer_call_fn_25055009
inputs_0
inputs_1
inputs_2
identityф
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_250535812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/2
Ё
 
+__inference_conv2d_2_layer_call_fn_25054832

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250534372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
H__inference_reshape_10_layer_call_and_return_conditional_losses_25055426

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
strided_slice/stack_2т
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
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј

+__inference_dense_24_layer_call_fn_25055314

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_250528952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ј

+__inference_dense_28_layer_call_fn_25055354

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_250528612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

ї
F__inference_dense_28_layer_call_and_return_conditional_losses_25052861

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

і
E__inference_dense_1_layer_call_and_return_conditional_losses_25053550

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ
i
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25052813

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і

*__inference_dense_1_layer_call_fn_25054974

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_250535502
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј

+__inference_dense_26_layer_call_fn_25055334

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_250528782
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
=
у
E__inference_decoder_layer_call_and_return_conditional_losses_25053022

inputs#
dense_23_25052845:  
dense_23_25052847: #
dense_28_25052862: 
dense_28_25052864:#
dense_26_25052879: 
dense_26_25052881:#
dense_24_25052896: 
dense_24_25052898:#
dense_29_25052913:0
dense_29_25052915:0#
dense_27_25052930:d
dense_27_25052932:d#
dense_25_25052947:
dense_25_25052949:
identity

identity_1

identity_2Ђ dense_23/StatefulPartitionedCallЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ dense_27/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinputsdense_23_25052845dense_23_25052847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_250528442"
 dense_23/StatefulPartitionedCallР
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_28_25052862dense_28_25052864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_250528612"
 dense_28/StatefulPartitionedCallР
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_26_25052879dense_26_25052881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_250528782"
 dense_26/StatefulPartitionedCallР
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_25052896dense_24_25052898*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_250528952"
 dense_24/StatefulPartitionedCallР
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_25052913dense_29_25052915*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_250529122"
 dense_29/StatefulPartitionedCallР
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_25052930dense_27_25052932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_250529292"
 dense_27/StatefulPartitionedCallР
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_25052947dense_25_25052949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_250529462"
 dense_25/StatefulPartitionedCall
reshape_12/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_12_layer_call_and_return_conditional_losses_250529662
reshape_12/PartitionedCall
reshape_11/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_250529822
reshape_11/PartitionedCall
reshape_10/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_250529962
reshape_10/PartitionedCall
softmax_2/PartitionedCallPartitionedCall#reshape_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_250530032
softmax_2/PartitionedCall
softmax_1/PartitionedCallPartitionedCall#reshape_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_250530102
softmax_1/PartitionedCallѓ
softmax/PartitionedCallPartitionedCall#reshape_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_250530172
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity"softmax_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity"softmax_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2У
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

ї
F__inference_dense_23_layer_call_and_return_conditional_losses_25052844

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

ї
F__inference_dense_29_layer_call_and_return_conditional_losses_25055405

inputs0
matmul_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
С
F
*__inference_softmax_layer_call_fn_25055479

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_250530172
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё=
ф
E__inference_decoder_layer_call_and_return_conditional_losses_25053415
input_6#
dense_23_25053371:  
dense_23_25053373: #
dense_28_25053376: 
dense_28_25053378:#
dense_26_25053381: 
dense_26_25053383:#
dense_24_25053386: 
dense_24_25053388:#
dense_29_25053391:0
dense_29_25053393:0#
dense_27_25053396:d
dense_27_25053398:d#
dense_25_25053401:
dense_25_25053403:
identity

identity_1

identity_2Ђ dense_23/StatefulPartitionedCallЂ dense_24/StatefulPartitionedCallЂ dense_25/StatefulPartitionedCallЂ dense_26/StatefulPartitionedCallЂ dense_27/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCall
 dense_23/StatefulPartitionedCallStatefulPartitionedCallinput_6dense_23_25053371dense_23_25053373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_250528442"
 dense_23/StatefulPartitionedCallР
 dense_28/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_28_25053376dense_28_25053378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_28_layer_call_and_return_conditional_losses_250528612"
 dense_28/StatefulPartitionedCallР
 dense_26/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_26_25053381dense_26_25053383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_26_layer_call_and_return_conditional_losses_250528782"
 dense_26/StatefulPartitionedCallР
 dense_24/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0dense_24_25053386dense_24_25053388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_24_layer_call_and_return_conditional_losses_250528952"
 dense_24/StatefulPartitionedCallР
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_25053391dense_29_25053393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_29_layer_call_and_return_conditional_losses_250529122"
 dense_29/StatefulPartitionedCallР
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_25053396dense_27_25053398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_250529292"
 dense_27/StatefulPartitionedCallР
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_25053401dense_25_25053403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_25_layer_call_and_return_conditional_losses_250529462"
 dense_25/StatefulPartitionedCall
reshape_12/PartitionedCallPartitionedCall)dense_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_12_layer_call_and_return_conditional_losses_250529662
reshape_12/PartitionedCall
reshape_11/PartitionedCallPartitionedCall)dense_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_11_layer_call_and_return_conditional_losses_250529822
reshape_11/PartitionedCall
reshape_10/PartitionedCallPartitionedCall)dense_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_10_layer_call_and_return_conditional_losses_250529962
reshape_10/PartitionedCall
softmax_2/PartitionedCallPartitionedCall#reshape_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_250530032
softmax_2/PartitionedCall
softmax_1/PartitionedCallPartitionedCall#reshape_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_250530102
softmax_1/PartitionedCallѓ
softmax/PartitionedCallPartitionedCall#reshape_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_softmax_layer_call_and_return_conditional_losses_250530172
softmax/PartitionedCall{
IdentityIdentity softmax/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity"softmax_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity"softmax_2/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2У
NoOpNoOp!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ : : : : : : : : : : : : : : 2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ 
!
_user_specified_name	input_6

ї
F__inference_dense_25_layer_call_and_return_conditional_losses_25055365

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ
ќ
-__inference_PASTA_CTAE_layer_call_fn_25054107
input_1
input_2
input_3!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:0 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18:

unknown_19: 

unknown_20:

unknown_21: 

unknown_22:

unknown_23:0

unknown_24:0

unknown_25:d

unknown_26:d

unknown_27:

unknown_28:
identity

identity_1

identity_2ЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*@
_read_only_resource_inputs"
 	
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_250539692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:XT
/
_output_shapes
:џџџџџџџџџ

!
_user_specified_name	input_2:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3

ї
F__inference_dense_26_layer_call_and_return_conditional_losses_25055325

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

ї
F__inference_dense_28_layer_call_and_return_conditional_losses_25055345

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_25052791

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
х
H
,__inference_softmax_1_layer_call_fn_25055489

inputs
identityа
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_softmax_1_layer_call_and_return_conditional_losses_250530102
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ
:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs


)__inference_conv2d_layer_call_fn_25054812

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_250534542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Н
g
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_25054842

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:џџџџџџџџџ	*
ksize
*
paddingSAME*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ	:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
і

*__inference_dense_2_layer_call_fn_25054994

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250535672
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
H__inference_reshape_11_layer_call_and_return_conditional_losses_25052982

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
strided_slice/stack_2т
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
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџd:O K
'
_output_shapes
:џџџџџџџџџd
 
_user_specified_nameinputs
Ё
 
+__inference_conv2d_1_layer_call_fn_25054892

inputs!
unknown:	
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_250535002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
х
a
E__inference_softmax_layer_call_and_return_conditional_losses_25055474

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
з
I
-__inference_reshape_12_layer_call_fn_25055469

inputs
identityб
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_reshape_12_layer_call_and_return_conditional_losses_250529662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ0:O K
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Й@
ю
!__inference__traced_save_25055616
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop4
0savev2_encoder_output_kernel_read_readvariableop2
.savev2_encoder_output_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameл
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*э
valueуBрB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЦ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesр
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop0savev2_encoder_output_kernel_read_readvariableop.savev2_encoder_output_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*Љ
_input_shapes
: :::::	::::::::::0 : :  : : :: :: ::::d:d:0:0: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:0 : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:0: 

_output_shapes
:0:

_output_shapes
: 
ј

+__inference_dense_27_layer_call_fn_25055394

inputs
unknown:d
	unknown_0:d
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџd*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_27_layer_call_and_return_conditional_losses_250529292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
о
N
2__inference_max_pooling2d_1_layer_call_fn_25054867

inputs
identityё
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_250528132
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ї
F__inference_dense_26_layer_call_and_return_conditional_losses_25052878

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

ї
F__inference_dense_29_layer_call_and_return_conditional_losses_25052912

inputs0
matmul_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ02

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
M

H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054191
input_1
input_2
input_3+
conv2d_2_25054112:
conv2d_2_25054114:)
conv2d_25054117:
conv2d_25054119:+
conv2d_3_25054124:
conv2d_3_25054126:+
conv2d_1_25054129:	
conv2d_1_25054131: 
dense_25054136:
dense_25054138:"
dense_1_25054141:
dense_1_25054143:"
dense_2_25054146:
dense_2_25054148:)
encoder_output_25054152:0 %
encoder_output_25054154: "
decoder_25054157:  
decoder_25054159: "
decoder_25054161: 
decoder_25054163:"
decoder_25054165: 
decoder_25054167:"
decoder_25054169: 
decoder_25054171:"
decoder_25054173:0
decoder_25054175:0"
decoder_25054177:d
decoder_25054179:d"
decoder_25054181:
decoder_25054183:
identity

identity_1

identity_2Ђconv2d/StatefulPartitionedCallЂ conv2d_1/StatefulPartitionedCallЂ conv2d_2/StatefulPartitionedCallЂ conv2d_3/StatefulPartitionedCallЂdecoder/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂ&encoder_output/StatefulPartitionedCallІ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_2_25054112conv2d_2_25054114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_250534372"
 conv2d_2/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_25054117conv2d_25054119*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_250534542 
conv2d/StatefulPartitionedCall
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_250534642!
max_pooling2d_1/PartitionedCall
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_250534702
max_pooling2d/PartitionedCallЧ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_3_25054124conv2d_3_25054126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_250534832"
 conv2d_3/StatefulPartitionedCallХ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_25054129conv2d_1_25054131*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_250535002"
 conv2d_1/StatefulPartitionedCallџ
flatten_1/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_250535122
flatten_1/PartitionedCallљ
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_250535202
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_25054136dense_25054138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_250535332
dense/StatefulPartitionedCallВ
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_25054141dense_1_25054143*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_250535502!
dense_1/StatefulPartitionedCallД
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_25054146dense_2_25054148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_250535672!
dense_2/StatefulPartitionedCallо
concatenate_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_concatenate_2_layer_call_and_return_conditional_losses_250535812
concatenate_2/PartitionedCallл
&encoder_output/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0encoder_output_25054152encoder_output_25054154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_encoder_output_layer_call_and_return_conditional_losses_250535932(
&encoder_output/StatefulPartitionedCallщ
decoder/StatefulPartitionedCallStatefulPartitionedCall/encoder_output/StatefulPartitionedCall:output:0decoder_25054157decoder_25054159decoder_25054161decoder_25054163decoder_25054165decoder_25054167decoder_25054169decoder_25054171decoder_25054173decoder_25054175decoder_25054177decoder_25054179decoder_25054181decoder_25054183*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_250530222!
decoder/StatefulPartitionedCall
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity(decoder/StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity(decoder/StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^decoder/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall'^encoder_output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2P
&encoder_output/StatefulPartitionedCall&encoder_output/StatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:XT
/
_output_shapes
:џџџџџџџџџ

!
_user_specified_name	input_2:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3
Ў

§
L__inference_encoder_output_layer_call_and_return_conditional_losses_25055019

inputs0
matmul_readvariableop_resource:0 -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
х
a
E__inference_softmax_layer_call_and_return_conditional_losses_25053017

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

d
H__inference_reshape_10_layer_call_and_return_conditional_losses_25052996

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
strided_slice/stack_2т
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
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ї
F__inference_dense_25_layer_call_and_return_conditional_losses_25052946

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
б~
§
$__inference__traced_restore_25055716
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_2_kernel:.
 assignvariableop_3_conv2d_2_bias:<
"assignvariableop_4_conv2d_1_kernel:	.
 assignvariableop_5_conv2d_1_bias:<
"assignvariableop_6_conv2d_3_kernel:.
 assignvariableop_7_conv2d_3_bias:1
assignvariableop_8_dense_kernel:+
assignvariableop_9_dense_bias:4
"assignvariableop_10_dense_1_kernel:.
 assignvariableop_11_dense_1_bias:4
"assignvariableop_12_dense_2_kernel:.
 assignvariableop_13_dense_2_bias:;
)assignvariableop_14_encoder_output_kernel:0 5
'assignvariableop_15_encoder_output_bias: 5
#assignvariableop_16_dense_23_kernel:  /
!assignvariableop_17_dense_23_bias: 5
#assignvariableop_18_dense_24_kernel: /
!assignvariableop_19_dense_24_bias:5
#assignvariableop_20_dense_26_kernel: /
!assignvariableop_21_dense_26_bias:5
#assignvariableop_22_dense_28_kernel: /
!assignvariableop_23_dense_28_bias:5
#assignvariableop_24_dense_25_kernel:/
!assignvariableop_25_dense_25_bias:5
#assignvariableop_26_dense_27_kernel:d/
!assignvariableop_27_dense_27_bias:d5
#assignvariableop_28_dense_29_kernel:0/
!assignvariableop_29_dense_29_bias:0
identity_31ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9с
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*э
valueуBрB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЧ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѓ
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ї
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѕ
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Є
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ђ
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Њ
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ј
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Њ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ј
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Б
AssignVariableOp_14AssignVariableOp)assignvariableop_14_encoder_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Џ
AssignVariableOp_15AssignVariableOp'assignvariableop_15_encoder_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ћ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_23_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Љ
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_23_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ћ
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_24_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Љ
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_24_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ћ
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_26_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Љ
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_26_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ћ
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_28_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Љ
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_28_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ћ
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_25_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Љ
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_25_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ћ
AssignVariableOp_26AssignVariableOp#assignvariableop_26_dense_27_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Љ
AssignVariableOp_27AssignVariableOp!assignvariableop_27_dense_27_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ћ
AssignVariableOp_28AssignVariableOp#assignvariableop_28_dense_29_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Љ
AssignVariableOp_29AssignVariableOp!assignvariableop_29_dense_29_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_299
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpђ
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30f
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_31к
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
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

d
H__inference_reshape_12_layer_call_and_return_conditional_losses_25052966

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
strided_slice/stack_2т
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
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ0:O K
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs
Ў

§
L__inference_encoder_output_layer_call_and_return_conditional_losses_25053593

inputs0
matmul_readvariableop_resource:0 -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:0 *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ0
 
_user_specified_nameinputs

і
E__inference_dense_1_layer_call_and_return_conditional_losses_25054965

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

џ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_25054823

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЛ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ї
F__inference_dense_27_layer_call_and_return_conditional_losses_25052929

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџd2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ї
F__inference_dense_24_layer_call_and_return_conditional_losses_25052895

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
э
џ
F__inference_conv2d_1_layer_call_and_return_conditional_losses_25054883

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
ј

+__inference_dense_23_layer_call_fn_25055294

inputs
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_250528442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

ї
F__inference_dense_24_layer_call_and_return_conditional_losses_25055305

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Њ
ќ
-__inference_PASTA_CTAE_layer_call_fn_25053700
input_1
input_2
input_3!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:0 

unknown_14: 

unknown_15:  

unknown_16: 

unknown_17: 

unknown_18:

unknown_19: 

unknown_20:

unknown_21: 

unknown_22:

unknown_23:0

unknown_24:0

unknown_25:d

unknown_26:d

unknown_27:

unknown_28:
identity

identity_1

identity_2ЂStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*@
_read_only_resource_inputs"
 	
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_250536332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:XT
/
_output_shapes
:џџџџџџџџџ

!
_user_specified_name	input_2:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3

є
C__inference_dense_layer_call_and_return_conditional_losses_25054945

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
L
0__inference_max_pooling2d_layer_call_fn_25054847

inputs
identityя
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_250527912
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

і
E__inference_dense_2_layer_call_and_return_conditional_losses_25054985

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
в
З
#__inference__wrapped_model_25052782
input_1
input_2
input_3L
2pasta_ctae_conv2d_2_conv2d_readvariableop_resource:A
3pasta_ctae_conv2d_2_biasadd_readvariableop_resource:J
0pasta_ctae_conv2d_conv2d_readvariableop_resource:?
1pasta_ctae_conv2d_biasadd_readvariableop_resource:L
2pasta_ctae_conv2d_3_conv2d_readvariableop_resource:A
3pasta_ctae_conv2d_3_biasadd_readvariableop_resource:L
2pasta_ctae_conv2d_1_conv2d_readvariableop_resource:	A
3pasta_ctae_conv2d_1_biasadd_readvariableop_resource:A
/pasta_ctae_dense_matmul_readvariableop_resource:>
0pasta_ctae_dense_biasadd_readvariableop_resource:C
1pasta_ctae_dense_1_matmul_readvariableop_resource:@
2pasta_ctae_dense_1_biasadd_readvariableop_resource:C
1pasta_ctae_dense_2_matmul_readvariableop_resource:@
2pasta_ctae_dense_2_biasadd_readvariableop_resource:J
8pasta_ctae_encoder_output_matmul_readvariableop_resource:0 G
9pasta_ctae_encoder_output_biasadd_readvariableop_resource: L
:pasta_ctae_decoder_dense_23_matmul_readvariableop_resource:  I
;pasta_ctae_decoder_dense_23_biasadd_readvariableop_resource: L
:pasta_ctae_decoder_dense_28_matmul_readvariableop_resource: I
;pasta_ctae_decoder_dense_28_biasadd_readvariableop_resource:L
:pasta_ctae_decoder_dense_26_matmul_readvariableop_resource: I
;pasta_ctae_decoder_dense_26_biasadd_readvariableop_resource:L
:pasta_ctae_decoder_dense_24_matmul_readvariableop_resource: I
;pasta_ctae_decoder_dense_24_biasadd_readvariableop_resource:L
:pasta_ctae_decoder_dense_29_matmul_readvariableop_resource:0I
;pasta_ctae_decoder_dense_29_biasadd_readvariableop_resource:0L
:pasta_ctae_decoder_dense_27_matmul_readvariableop_resource:dI
;pasta_ctae_decoder_dense_27_biasadd_readvariableop_resource:dL
:pasta_ctae_decoder_dense_25_matmul_readvariableop_resource:I
;pasta_ctae_decoder_dense_25_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђ(PASTA_CTAE/conv2d/BiasAdd/ReadVariableOpЂ'PASTA_CTAE/conv2d/Conv2D/ReadVariableOpЂ*PASTA_CTAE/conv2d_1/BiasAdd/ReadVariableOpЂ)PASTA_CTAE/conv2d_1/Conv2D/ReadVariableOpЂ*PASTA_CTAE/conv2d_2/BiasAdd/ReadVariableOpЂ)PASTA_CTAE/conv2d_2/Conv2D/ReadVariableOpЂ*PASTA_CTAE/conv2d_3/BiasAdd/ReadVariableOpЂ)PASTA_CTAE/conv2d_3/Conv2D/ReadVariableOpЂ2PASTA_CTAE/decoder/dense_23/BiasAdd/ReadVariableOpЂ1PASTA_CTAE/decoder/dense_23/MatMul/ReadVariableOpЂ2PASTA_CTAE/decoder/dense_24/BiasAdd/ReadVariableOpЂ1PASTA_CTAE/decoder/dense_24/MatMul/ReadVariableOpЂ2PASTA_CTAE/decoder/dense_25/BiasAdd/ReadVariableOpЂ1PASTA_CTAE/decoder/dense_25/MatMul/ReadVariableOpЂ2PASTA_CTAE/decoder/dense_26/BiasAdd/ReadVariableOpЂ1PASTA_CTAE/decoder/dense_26/MatMul/ReadVariableOpЂ2PASTA_CTAE/decoder/dense_27/BiasAdd/ReadVariableOpЂ1PASTA_CTAE/decoder/dense_27/MatMul/ReadVariableOpЂ2PASTA_CTAE/decoder/dense_28/BiasAdd/ReadVariableOpЂ1PASTA_CTAE/decoder/dense_28/MatMul/ReadVariableOpЂ2PASTA_CTAE/decoder/dense_29/BiasAdd/ReadVariableOpЂ1PASTA_CTAE/decoder/dense_29/MatMul/ReadVariableOpЂ'PASTA_CTAE/dense/BiasAdd/ReadVariableOpЂ&PASTA_CTAE/dense/MatMul/ReadVariableOpЂ)PASTA_CTAE/dense_1/BiasAdd/ReadVariableOpЂ(PASTA_CTAE/dense_1/MatMul/ReadVariableOpЂ)PASTA_CTAE/dense_2/BiasAdd/ReadVariableOpЂ(PASTA_CTAE/dense_2/MatMul/ReadVariableOpЂ0PASTA_CTAE/encoder_output/BiasAdd/ReadVariableOpЂ/PASTA_CTAE/encoder_output/MatMul/ReadVariableOpб
)PASTA_CTAE/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2pasta_ctae_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)PASTA_CTAE/conv2d_2/Conv2D/ReadVariableOpј
PASTA_CTAE/conv2d_2/Conv2DConv2Dinput_31PASTA_CTAE/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW*
paddingVALID*
strides
2
PASTA_CTAE/conv2d_2/Conv2DШ
*PASTA_CTAE/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3pasta_ctae_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*PASTA_CTAE/conv2d_2/BiasAdd/ReadVariableOpя
PASTA_CTAE/conv2d_2/BiasAddBiasAdd#PASTA_CTAE/conv2d_2/Conv2D:output:02PASTA_CTAE/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
data_formatNCHW2
PASTA_CTAE/conv2d_2/BiasAdd
PASTA_CTAE/conv2d_2/ReluRelu$PASTA_CTAE/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/conv2d_2/ReluЫ
'PASTA_CTAE/conv2d/Conv2D/ReadVariableOpReadVariableOp0pasta_ctae_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'PASTA_CTAE/conv2d/Conv2D/ReadVariableOpђ
PASTA_CTAE/conv2d/Conv2DConv2Dinput_2/PASTA_CTAE/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW*
paddingVALID*
strides
2
PASTA_CTAE/conv2d/Conv2DТ
(PASTA_CTAE/conv2d/BiasAdd/ReadVariableOpReadVariableOp1pasta_ctae_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(PASTA_CTAE/conv2d/BiasAdd/ReadVariableOpч
PASTA_CTAE/conv2d/BiasAddBiasAdd!PASTA_CTAE/conv2d/Conv2D:output:00PASTA_CTAE/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ	*
data_formatNCHW2
PASTA_CTAE/conv2d/BiasAdd
PASTA_CTAE/conv2d/ReluRelu"PASTA_CTAE/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ	2
PASTA_CTAE/conv2d/Reluч
"PASTA_CTAE/max_pooling2d_1/MaxPoolMaxPool&PASTA_CTAE/conv2d_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2$
"PASTA_CTAE/max_pooling2d_1/MaxPoolс
 PASTA_CTAE/max_pooling2d/MaxPoolMaxPool$PASTA_CTAE/conv2d/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ	*
ksize
*
paddingSAME*
strides
2"
 PASTA_CTAE/max_pooling2d/MaxPoolб
)PASTA_CTAE/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2pasta_ctae_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)PASTA_CTAE/conv2d_3/Conv2D/ReadVariableOp
PASTA_CTAE/conv2d_3/Conv2DConv2D+PASTA_CTAE/max_pooling2d_1/MaxPool:output:01PASTA_CTAE/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
PASTA_CTAE/conv2d_3/Conv2DШ
*PASTA_CTAE/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3pasta_ctae_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*PASTA_CTAE/conv2d_3/BiasAdd/ReadVariableOpи
PASTA_CTAE/conv2d_3/BiasAddBiasAdd#PASTA_CTAE/conv2d_3/Conv2D:output:02PASTA_CTAE/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/conv2d_3/BiasAdd
PASTA_CTAE/conv2d_3/ReluRelu$PASTA_CTAE/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/conv2d_3/Reluб
)PASTA_CTAE/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2pasta_ctae_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)PASTA_CTAE/conv2d_1/Conv2D/ReadVariableOp
PASTA_CTAE/conv2d_1/Conv2DConv2D)PASTA_CTAE/max_pooling2d/MaxPool:output:01PASTA_CTAE/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
PASTA_CTAE/conv2d_1/Conv2DШ
*PASTA_CTAE/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3pasta_ctae_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*PASTA_CTAE/conv2d_1/BiasAdd/ReadVariableOpи
PASTA_CTAE/conv2d_1/BiasAddBiasAdd#PASTA_CTAE/conv2d_1/Conv2D:output:02PASTA_CTAE/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/conv2d_1/BiasAdd
PASTA_CTAE/conv2d_1/ReluRelu$PASTA_CTAE/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/conv2d_1/Relu
PASTA_CTAE/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
PASTA_CTAE/flatten_1/ConstЦ
PASTA_CTAE/flatten_1/ReshapeReshape&PASTA_CTAE/conv2d_3/Relu:activations:0#PASTA_CTAE/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/flatten_1/Reshape
PASTA_CTAE/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
PASTA_CTAE/flatten/ConstР
PASTA_CTAE/flatten/ReshapeReshape&PASTA_CTAE/conv2d_1/Relu:activations:0!PASTA_CTAE/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/flatten/ReshapeР
&PASTA_CTAE/dense/MatMul/ReadVariableOpReadVariableOp/pasta_ctae_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&PASTA_CTAE/dense/MatMul/ReadVariableOpЇ
PASTA_CTAE/dense/MatMulMatMulinput_1.PASTA_CTAE/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense/MatMulП
'PASTA_CTAE/dense/BiasAdd/ReadVariableOpReadVariableOp0pasta_ctae_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'PASTA_CTAE/dense/BiasAdd/ReadVariableOpХ
PASTA_CTAE/dense/BiasAddBiasAdd!PASTA_CTAE/dense/MatMul:product:0/PASTA_CTAE/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense/BiasAdd
PASTA_CTAE/dense/ReluRelu!PASTA_CTAE/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense/ReluЦ
(PASTA_CTAE/dense_1/MatMul/ReadVariableOpReadVariableOp1pasta_ctae_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(PASTA_CTAE/dense_1/MatMul/ReadVariableOpЩ
PASTA_CTAE/dense_1/MatMulMatMul#PASTA_CTAE/flatten/Reshape:output:00PASTA_CTAE/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense_1/MatMulХ
)PASTA_CTAE/dense_1/BiasAdd/ReadVariableOpReadVariableOp2pasta_ctae_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)PASTA_CTAE/dense_1/BiasAdd/ReadVariableOpЭ
PASTA_CTAE/dense_1/BiasAddBiasAdd#PASTA_CTAE/dense_1/MatMul:product:01PASTA_CTAE/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense_1/BiasAdd
PASTA_CTAE/dense_1/ReluRelu#PASTA_CTAE/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense_1/ReluЦ
(PASTA_CTAE/dense_2/MatMul/ReadVariableOpReadVariableOp1pasta_ctae_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(PASTA_CTAE/dense_2/MatMul/ReadVariableOpЫ
PASTA_CTAE/dense_2/MatMulMatMul%PASTA_CTAE/flatten_1/Reshape:output:00PASTA_CTAE/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense_2/MatMulХ
)PASTA_CTAE/dense_2/BiasAdd/ReadVariableOpReadVariableOp2pasta_ctae_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)PASTA_CTAE/dense_2/BiasAdd/ReadVariableOpЭ
PASTA_CTAE/dense_2/BiasAddBiasAdd#PASTA_CTAE/dense_2/MatMul:product:01PASTA_CTAE/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense_2/BiasAdd
PASTA_CTAE/dense_2/ReluRelu#PASTA_CTAE/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
PASTA_CTAE/dense_2/Relu
$PASTA_CTAE/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$PASTA_CTAE/concatenate_2/concat/axisЋ
PASTA_CTAE/concatenate_2/concatConcatV2#PASTA_CTAE/dense/Relu:activations:0%PASTA_CTAE/dense_1/Relu:activations:0%PASTA_CTAE/dense_2/Relu:activations:0-PASTA_CTAE/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ02!
PASTA_CTAE/concatenate_2/concatл
/PASTA_CTAE/encoder_output/MatMul/ReadVariableOpReadVariableOp8pasta_ctae_encoder_output_matmul_readvariableop_resource*
_output_shapes

:0 *
dtype021
/PASTA_CTAE/encoder_output/MatMul/ReadVariableOpу
 PASTA_CTAE/encoder_output/MatMulMatMul(PASTA_CTAE/concatenate_2/concat:output:07PASTA_CTAE/encoder_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 PASTA_CTAE/encoder_output/MatMulк
0PASTA_CTAE/encoder_output/BiasAdd/ReadVariableOpReadVariableOp9pasta_ctae_encoder_output_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0PASTA_CTAE/encoder_output/BiasAdd/ReadVariableOpщ
!PASTA_CTAE/encoder_output/BiasAddBiasAdd*PASTA_CTAE/encoder_output/MatMul:product:08PASTA_CTAE/encoder_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2#
!PASTA_CTAE/encoder_output/BiasAddс
1PASTA_CTAE/decoder/dense_23/MatMul/ReadVariableOpReadVariableOp:pasta_ctae_decoder_dense_23_matmul_readvariableop_resource*
_output_shapes

:  *
dtype023
1PASTA_CTAE/decoder/dense_23/MatMul/ReadVariableOpы
"PASTA_CTAE/decoder/dense_23/MatMulMatMul*PASTA_CTAE/encoder_output/BiasAdd:output:09PASTA_CTAE/decoder/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2$
"PASTA_CTAE/decoder/dense_23/MatMulр
2PASTA_CTAE/decoder/dense_23/BiasAdd/ReadVariableOpReadVariableOp;pasta_ctae_decoder_dense_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2PASTA_CTAE/decoder/dense_23/BiasAdd/ReadVariableOpё
#PASTA_CTAE/decoder/dense_23/BiasAddBiasAdd,PASTA_CTAE/decoder/dense_23/MatMul:product:0:PASTA_CTAE/decoder/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 2%
#PASTA_CTAE/decoder/dense_23/BiasAddЌ
 PASTA_CTAE/decoder/dense_23/ReluRelu,PASTA_CTAE/decoder/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 2"
 PASTA_CTAE/decoder/dense_23/Reluс
1PASTA_CTAE/decoder/dense_28/MatMul/ReadVariableOpReadVariableOp:pasta_ctae_decoder_dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype023
1PASTA_CTAE/decoder/dense_28/MatMul/ReadVariableOpя
"PASTA_CTAE/decoder/dense_28/MatMulMatMul.PASTA_CTAE/decoder/dense_23/Relu:activations:09PASTA_CTAE/decoder/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"PASTA_CTAE/decoder/dense_28/MatMulр
2PASTA_CTAE/decoder/dense_28/BiasAdd/ReadVariableOpReadVariableOp;pasta_ctae_decoder_dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2PASTA_CTAE/decoder/dense_28/BiasAdd/ReadVariableOpё
#PASTA_CTAE/decoder/dense_28/BiasAddBiasAdd,PASTA_CTAE/decoder/dense_28/MatMul:product:0:PASTA_CTAE/decoder/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#PASTA_CTAE/decoder/dense_28/BiasAddЌ
 PASTA_CTAE/decoder/dense_28/ReluRelu,PASTA_CTAE/decoder/dense_28/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 PASTA_CTAE/decoder/dense_28/Reluс
1PASTA_CTAE/decoder/dense_26/MatMul/ReadVariableOpReadVariableOp:pasta_ctae_decoder_dense_26_matmul_readvariableop_resource*
_output_shapes

: *
dtype023
1PASTA_CTAE/decoder/dense_26/MatMul/ReadVariableOpя
"PASTA_CTAE/decoder/dense_26/MatMulMatMul.PASTA_CTAE/decoder/dense_23/Relu:activations:09PASTA_CTAE/decoder/dense_26/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"PASTA_CTAE/decoder/dense_26/MatMulр
2PASTA_CTAE/decoder/dense_26/BiasAdd/ReadVariableOpReadVariableOp;pasta_ctae_decoder_dense_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2PASTA_CTAE/decoder/dense_26/BiasAdd/ReadVariableOpё
#PASTA_CTAE/decoder/dense_26/BiasAddBiasAdd,PASTA_CTAE/decoder/dense_26/MatMul:product:0:PASTA_CTAE/decoder/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#PASTA_CTAE/decoder/dense_26/BiasAddЌ
 PASTA_CTAE/decoder/dense_26/ReluRelu,PASTA_CTAE/decoder/dense_26/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 PASTA_CTAE/decoder/dense_26/Reluс
1PASTA_CTAE/decoder/dense_24/MatMul/ReadVariableOpReadVariableOp:pasta_ctae_decoder_dense_24_matmul_readvariableop_resource*
_output_shapes

: *
dtype023
1PASTA_CTAE/decoder/dense_24/MatMul/ReadVariableOpя
"PASTA_CTAE/decoder/dense_24/MatMulMatMul.PASTA_CTAE/decoder/dense_23/Relu:activations:09PASTA_CTAE/decoder/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"PASTA_CTAE/decoder/dense_24/MatMulр
2PASTA_CTAE/decoder/dense_24/BiasAdd/ReadVariableOpReadVariableOp;pasta_ctae_decoder_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2PASTA_CTAE/decoder/dense_24/BiasAdd/ReadVariableOpё
#PASTA_CTAE/decoder/dense_24/BiasAddBiasAdd,PASTA_CTAE/decoder/dense_24/MatMul:product:0:PASTA_CTAE/decoder/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#PASTA_CTAE/decoder/dense_24/BiasAddЌ
 PASTA_CTAE/decoder/dense_24/ReluRelu,PASTA_CTAE/decoder/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 PASTA_CTAE/decoder/dense_24/Reluс
1PASTA_CTAE/decoder/dense_29/MatMul/ReadVariableOpReadVariableOp:pasta_ctae_decoder_dense_29_matmul_readvariableop_resource*
_output_shapes

:0*
dtype023
1PASTA_CTAE/decoder/dense_29/MatMul/ReadVariableOpя
"PASTA_CTAE/decoder/dense_29/MatMulMatMul.PASTA_CTAE/decoder/dense_28/Relu:activations:09PASTA_CTAE/decoder/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02$
"PASTA_CTAE/decoder/dense_29/MatMulр
2PASTA_CTAE/decoder/dense_29/BiasAdd/ReadVariableOpReadVariableOp;pasta_ctae_decoder_dense_29_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype024
2PASTA_CTAE/decoder/dense_29/BiasAdd/ReadVariableOpё
#PASTA_CTAE/decoder/dense_29/BiasAddBiasAdd,PASTA_CTAE/decoder/dense_29/MatMul:product:0:PASTA_CTAE/decoder/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ02%
#PASTA_CTAE/decoder/dense_29/BiasAddЌ
 PASTA_CTAE/decoder/dense_29/ReluRelu,PASTA_CTAE/decoder/dense_29/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ02"
 PASTA_CTAE/decoder/dense_29/Reluс
1PASTA_CTAE/decoder/dense_27/MatMul/ReadVariableOpReadVariableOp:pasta_ctae_decoder_dense_27_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1PASTA_CTAE/decoder/dense_27/MatMul/ReadVariableOpя
"PASTA_CTAE/decoder/dense_27/MatMulMatMul.PASTA_CTAE/decoder/dense_26/Relu:activations:09PASTA_CTAE/decoder/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2$
"PASTA_CTAE/decoder/dense_27/MatMulр
2PASTA_CTAE/decoder/dense_27/BiasAdd/ReadVariableOpReadVariableOp;pasta_ctae_decoder_dense_27_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype024
2PASTA_CTAE/decoder/dense_27/BiasAdd/ReadVariableOpё
#PASTA_CTAE/decoder/dense_27/BiasAddBiasAdd,PASTA_CTAE/decoder/dense_27/MatMul:product:0:PASTA_CTAE/decoder/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd2%
#PASTA_CTAE/decoder/dense_27/BiasAddЌ
 PASTA_CTAE/decoder/dense_27/ReluRelu,PASTA_CTAE/decoder/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2"
 PASTA_CTAE/decoder/dense_27/Reluс
1PASTA_CTAE/decoder/dense_25/MatMul/ReadVariableOpReadVariableOp:pasta_ctae_decoder_dense_25_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1PASTA_CTAE/decoder/dense_25/MatMul/ReadVariableOpя
"PASTA_CTAE/decoder/dense_25/MatMulMatMul.PASTA_CTAE/decoder/dense_24/Relu:activations:09PASTA_CTAE/decoder/dense_25/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"PASTA_CTAE/decoder/dense_25/MatMulр
2PASTA_CTAE/decoder/dense_25/BiasAdd/ReadVariableOpReadVariableOp;pasta_ctae_decoder_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2PASTA_CTAE/decoder/dense_25/BiasAdd/ReadVariableOpё
#PASTA_CTAE/decoder/dense_25/BiasAddBiasAdd,PASTA_CTAE/decoder/dense_25/MatMul:product:0:PASTA_CTAE/decoder/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#PASTA_CTAE/decoder/dense_25/BiasAddЌ
 PASTA_CTAE/decoder/dense_25/ReluRelu,PASTA_CTAE/decoder/dense_25/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 PASTA_CTAE/decoder/dense_25/ReluЈ
#PASTA_CTAE/decoder/reshape_12/ShapeShape.PASTA_CTAE/decoder/dense_29/Relu:activations:0*
T0*
_output_shapes
:2%
#PASTA_CTAE/decoder/reshape_12/ShapeА
1PASTA_CTAE/decoder/reshape_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1PASTA_CTAE/decoder/reshape_12/strided_slice/stackД
3PASTA_CTAE/decoder/reshape_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3PASTA_CTAE/decoder/reshape_12/strided_slice/stack_1Д
3PASTA_CTAE/decoder/reshape_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3PASTA_CTAE/decoder/reshape_12/strided_slice/stack_2
+PASTA_CTAE/decoder/reshape_12/strided_sliceStridedSlice,PASTA_CTAE/decoder/reshape_12/Shape:output:0:PASTA_CTAE/decoder/reshape_12/strided_slice/stack:output:0<PASTA_CTAE/decoder/reshape_12/strided_slice/stack_1:output:0<PASTA_CTAE/decoder/reshape_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+PASTA_CTAE/decoder/reshape_12/strided_slice 
-PASTA_CTAE/decoder/reshape_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-PASTA_CTAE/decoder/reshape_12/Reshape/shape/1 
-PASTA_CTAE/decoder/reshape_12/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-PASTA_CTAE/decoder/reshape_12/Reshape/shape/2 
-PASTA_CTAE/decoder/reshape_12/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-PASTA_CTAE/decoder/reshape_12/Reshape/shape/3ю
+PASTA_CTAE/decoder/reshape_12/Reshape/shapePack4PASTA_CTAE/decoder/reshape_12/strided_slice:output:06PASTA_CTAE/decoder/reshape_12/Reshape/shape/1:output:06PASTA_CTAE/decoder/reshape_12/Reshape/shape/2:output:06PASTA_CTAE/decoder/reshape_12/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+PASTA_CTAE/decoder/reshape_12/Reshape/shapeљ
%PASTA_CTAE/decoder/reshape_12/ReshapeReshape.PASTA_CTAE/decoder/dense_29/Relu:activations:04PASTA_CTAE/decoder/reshape_12/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2'
%PASTA_CTAE/decoder/reshape_12/ReshapeЈ
#PASTA_CTAE/decoder/reshape_11/ShapeShape.PASTA_CTAE/decoder/dense_27/Relu:activations:0*
T0*
_output_shapes
:2%
#PASTA_CTAE/decoder/reshape_11/ShapeА
1PASTA_CTAE/decoder/reshape_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1PASTA_CTAE/decoder/reshape_11/strided_slice/stackД
3PASTA_CTAE/decoder/reshape_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3PASTA_CTAE/decoder/reshape_11/strided_slice/stack_1Д
3PASTA_CTAE/decoder/reshape_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3PASTA_CTAE/decoder/reshape_11/strided_slice/stack_2
+PASTA_CTAE/decoder/reshape_11/strided_sliceStridedSlice,PASTA_CTAE/decoder/reshape_11/Shape:output:0:PASTA_CTAE/decoder/reshape_11/strided_slice/stack:output:0<PASTA_CTAE/decoder/reshape_11/strided_slice/stack_1:output:0<PASTA_CTAE/decoder/reshape_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+PASTA_CTAE/decoder/reshape_11/strided_slice 
-PASTA_CTAE/decoder/reshape_11/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-PASTA_CTAE/decoder/reshape_11/Reshape/shape/1 
-PASTA_CTAE/decoder/reshape_11/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-PASTA_CTAE/decoder/reshape_11/Reshape/shape/2 
-PASTA_CTAE/decoder/reshape_11/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :
2/
-PASTA_CTAE/decoder/reshape_11/Reshape/shape/3ю
+PASTA_CTAE/decoder/reshape_11/Reshape/shapePack4PASTA_CTAE/decoder/reshape_11/strided_slice:output:06PASTA_CTAE/decoder/reshape_11/Reshape/shape/1:output:06PASTA_CTAE/decoder/reshape_11/Reshape/shape/2:output:06PASTA_CTAE/decoder/reshape_11/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+PASTA_CTAE/decoder/reshape_11/Reshape/shapeљ
%PASTA_CTAE/decoder/reshape_11/ReshapeReshape.PASTA_CTAE/decoder/dense_27/Relu:activations:04PASTA_CTAE/decoder/reshape_11/Reshape/shape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2'
%PASTA_CTAE/decoder/reshape_11/ReshapeЈ
#PASTA_CTAE/decoder/reshape_10/ShapeShape.PASTA_CTAE/decoder/dense_25/Relu:activations:0*
T0*
_output_shapes
:2%
#PASTA_CTAE/decoder/reshape_10/ShapeА
1PASTA_CTAE/decoder/reshape_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1PASTA_CTAE/decoder/reshape_10/strided_slice/stackД
3PASTA_CTAE/decoder/reshape_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3PASTA_CTAE/decoder/reshape_10/strided_slice/stack_1Д
3PASTA_CTAE/decoder/reshape_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3PASTA_CTAE/decoder/reshape_10/strided_slice/stack_2
+PASTA_CTAE/decoder/reshape_10/strided_sliceStridedSlice,PASTA_CTAE/decoder/reshape_10/Shape:output:0:PASTA_CTAE/decoder/reshape_10/strided_slice/stack:output:0<PASTA_CTAE/decoder/reshape_10/strided_slice/stack_1:output:0<PASTA_CTAE/decoder/reshape_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+PASTA_CTAE/decoder/reshape_10/strided_slice 
-PASTA_CTAE/decoder/reshape_10/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-PASTA_CTAE/decoder/reshape_10/Reshape/shape/1ў
+PASTA_CTAE/decoder/reshape_10/Reshape/shapePack4PASTA_CTAE/decoder/reshape_10/strided_slice:output:06PASTA_CTAE/decoder/reshape_10/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2-
+PASTA_CTAE/decoder/reshape_10/Reshape/shapeё
%PASTA_CTAE/decoder/reshape_10/ReshapeReshape.PASTA_CTAE/decoder/dense_25/Relu:activations:04PASTA_CTAE/decoder/reshape_10/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2'
%PASTA_CTAE/decoder/reshape_10/ReshapeС
$PASTA_CTAE/decoder/softmax_2/SoftmaxSoftmax.PASTA_CTAE/decoder/reshape_12/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2&
$PASTA_CTAE/decoder/softmax_2/SoftmaxС
$PASTA_CTAE/decoder/softmax_1/SoftmaxSoftmax.PASTA_CTAE/decoder/reshape_11/Reshape:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
2&
$PASTA_CTAE/decoder/softmax_1/SoftmaxЕ
"PASTA_CTAE/decoder/softmax/SoftmaxSoftmax.PASTA_CTAE/decoder/reshape_10/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"PASTA_CTAE/decoder/softmax/Softmax
IdentityIdentity,PASTA_CTAE/decoder/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.PASTA_CTAE/decoder/softmax_1/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity.PASTA_CTAE/decoder/softmax_2/Softmax:softmax:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity_2ѓ
NoOpNoOp)^PASTA_CTAE/conv2d/BiasAdd/ReadVariableOp(^PASTA_CTAE/conv2d/Conv2D/ReadVariableOp+^PASTA_CTAE/conv2d_1/BiasAdd/ReadVariableOp*^PASTA_CTAE/conv2d_1/Conv2D/ReadVariableOp+^PASTA_CTAE/conv2d_2/BiasAdd/ReadVariableOp*^PASTA_CTAE/conv2d_2/Conv2D/ReadVariableOp+^PASTA_CTAE/conv2d_3/BiasAdd/ReadVariableOp*^PASTA_CTAE/conv2d_3/Conv2D/ReadVariableOp3^PASTA_CTAE/decoder/dense_23/BiasAdd/ReadVariableOp2^PASTA_CTAE/decoder/dense_23/MatMul/ReadVariableOp3^PASTA_CTAE/decoder/dense_24/BiasAdd/ReadVariableOp2^PASTA_CTAE/decoder/dense_24/MatMul/ReadVariableOp3^PASTA_CTAE/decoder/dense_25/BiasAdd/ReadVariableOp2^PASTA_CTAE/decoder/dense_25/MatMul/ReadVariableOp3^PASTA_CTAE/decoder/dense_26/BiasAdd/ReadVariableOp2^PASTA_CTAE/decoder/dense_26/MatMul/ReadVariableOp3^PASTA_CTAE/decoder/dense_27/BiasAdd/ReadVariableOp2^PASTA_CTAE/decoder/dense_27/MatMul/ReadVariableOp3^PASTA_CTAE/decoder/dense_28/BiasAdd/ReadVariableOp2^PASTA_CTAE/decoder/dense_28/MatMul/ReadVariableOp3^PASTA_CTAE/decoder/dense_29/BiasAdd/ReadVariableOp2^PASTA_CTAE/decoder/dense_29/MatMul/ReadVariableOp(^PASTA_CTAE/dense/BiasAdd/ReadVariableOp'^PASTA_CTAE/dense/MatMul/ReadVariableOp*^PASTA_CTAE/dense_1/BiasAdd/ReadVariableOp)^PASTA_CTAE/dense_1/MatMul/ReadVariableOp*^PASTA_CTAE/dense_2/BiasAdd/ReadVariableOp)^PASTA_CTAE/dense_2/MatMul/ReadVariableOp1^PASTA_CTAE/encoder_output/BiasAdd/ReadVariableOp0^PASTA_CTAE/encoder_output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(PASTA_CTAE/conv2d/BiasAdd/ReadVariableOp(PASTA_CTAE/conv2d/BiasAdd/ReadVariableOp2R
'PASTA_CTAE/conv2d/Conv2D/ReadVariableOp'PASTA_CTAE/conv2d/Conv2D/ReadVariableOp2X
*PASTA_CTAE/conv2d_1/BiasAdd/ReadVariableOp*PASTA_CTAE/conv2d_1/BiasAdd/ReadVariableOp2V
)PASTA_CTAE/conv2d_1/Conv2D/ReadVariableOp)PASTA_CTAE/conv2d_1/Conv2D/ReadVariableOp2X
*PASTA_CTAE/conv2d_2/BiasAdd/ReadVariableOp*PASTA_CTAE/conv2d_2/BiasAdd/ReadVariableOp2V
)PASTA_CTAE/conv2d_2/Conv2D/ReadVariableOp)PASTA_CTAE/conv2d_2/Conv2D/ReadVariableOp2X
*PASTA_CTAE/conv2d_3/BiasAdd/ReadVariableOp*PASTA_CTAE/conv2d_3/BiasAdd/ReadVariableOp2V
)PASTA_CTAE/conv2d_3/Conv2D/ReadVariableOp)PASTA_CTAE/conv2d_3/Conv2D/ReadVariableOp2h
2PASTA_CTAE/decoder/dense_23/BiasAdd/ReadVariableOp2PASTA_CTAE/decoder/dense_23/BiasAdd/ReadVariableOp2f
1PASTA_CTAE/decoder/dense_23/MatMul/ReadVariableOp1PASTA_CTAE/decoder/dense_23/MatMul/ReadVariableOp2h
2PASTA_CTAE/decoder/dense_24/BiasAdd/ReadVariableOp2PASTA_CTAE/decoder/dense_24/BiasAdd/ReadVariableOp2f
1PASTA_CTAE/decoder/dense_24/MatMul/ReadVariableOp1PASTA_CTAE/decoder/dense_24/MatMul/ReadVariableOp2h
2PASTA_CTAE/decoder/dense_25/BiasAdd/ReadVariableOp2PASTA_CTAE/decoder/dense_25/BiasAdd/ReadVariableOp2f
1PASTA_CTAE/decoder/dense_25/MatMul/ReadVariableOp1PASTA_CTAE/decoder/dense_25/MatMul/ReadVariableOp2h
2PASTA_CTAE/decoder/dense_26/BiasAdd/ReadVariableOp2PASTA_CTAE/decoder/dense_26/BiasAdd/ReadVariableOp2f
1PASTA_CTAE/decoder/dense_26/MatMul/ReadVariableOp1PASTA_CTAE/decoder/dense_26/MatMul/ReadVariableOp2h
2PASTA_CTAE/decoder/dense_27/BiasAdd/ReadVariableOp2PASTA_CTAE/decoder/dense_27/BiasAdd/ReadVariableOp2f
1PASTA_CTAE/decoder/dense_27/MatMul/ReadVariableOp1PASTA_CTAE/decoder/dense_27/MatMul/ReadVariableOp2h
2PASTA_CTAE/decoder/dense_28/BiasAdd/ReadVariableOp2PASTA_CTAE/decoder/dense_28/BiasAdd/ReadVariableOp2f
1PASTA_CTAE/decoder/dense_28/MatMul/ReadVariableOp1PASTA_CTAE/decoder/dense_28/MatMul/ReadVariableOp2h
2PASTA_CTAE/decoder/dense_29/BiasAdd/ReadVariableOp2PASTA_CTAE/decoder/dense_29/BiasAdd/ReadVariableOp2f
1PASTA_CTAE/decoder/dense_29/MatMul/ReadVariableOp1PASTA_CTAE/decoder/dense_29/MatMul/ReadVariableOp2R
'PASTA_CTAE/dense/BiasAdd/ReadVariableOp'PASTA_CTAE/dense/BiasAdd/ReadVariableOp2P
&PASTA_CTAE/dense/MatMul/ReadVariableOp&PASTA_CTAE/dense/MatMul/ReadVariableOp2V
)PASTA_CTAE/dense_1/BiasAdd/ReadVariableOp)PASTA_CTAE/dense_1/BiasAdd/ReadVariableOp2T
(PASTA_CTAE/dense_1/MatMul/ReadVariableOp(PASTA_CTAE/dense_1/MatMul/ReadVariableOp2V
)PASTA_CTAE/dense_2/BiasAdd/ReadVariableOp)PASTA_CTAE/dense_2/BiasAdd/ReadVariableOp2T
(PASTA_CTAE/dense_2/MatMul/ReadVariableOp(PASTA_CTAE/dense_2/MatMul/ReadVariableOp2d
0PASTA_CTAE/encoder_output/BiasAdd/ReadVariableOp0PASTA_CTAE/encoder_output/BiasAdd/ReadVariableOp2b
/PASTA_CTAE/encoder_output/MatMul/ReadVariableOp/PASTA_CTAE/encoder_output/MatMul/ReadVariableOp:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1:XT
/
_output_shapes
:џџџџџџџџџ

!
_user_specified_name	input_2:XT
/
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3
І
ў
*__inference_decoder_layer_call_fn_25055274

inputs
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6:
	unknown_7:0
	unknown_8:0
	unknown_9:d

unknown_10:d

unknown_11:

unknown_12:
identity

identity_1

identity_2ЂStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:џџџџџџџџџ:џџџџџџџџџ
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoder_layer_call_and_return_conditional_losses_250532492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

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
/:џџџџџџџџџ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
э
џ
F__inference_conv2d_1_layer_call_and_return_conditional_losses_25053500

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
б
F
*__inference_flatten_layer_call_fn_25054923

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_250535202
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ

(__inference_dense_layer_call_fn_25054954

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_250535332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Т
serving_defaultЎ
;
input_10
serving_default_input_1:0џџџџџџџџџ
C
input_28
serving_default_input_2:0џџџџџџџџџ

C
input_38
serving_default_input_3:0џџџџџџџџџ;
decoder0
StatefulPartitionedCall:0џџџџџџџџџE
	decoder_18
StatefulPartitionedCall:1џџџџџџџџџ
E
	decoder_28
StatefulPartitionedCall:2џџџџџџџџџtensorflow/serving/predict:рЈ
Ѕ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
trainable_variables
regularization_losses
	variables
	keras_api

signatures
М_default_save_signature
+Н&call_and_return_all_conditional_losses
О__call__"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Н

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"
_tf_keras_layer
Н

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"
_tf_keras_layer
Ї
#trainable_variables
$regularization_losses
%	variables
&	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"
_tf_keras_layer
Ї
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"
_tf_keras_layer
Н

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"
_tf_keras_layer
Н

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"
_tf_keras_layer
"
_tf_keras_input_layer
Ї
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"
_tf_keras_layer
Ї
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"
_tf_keras_layer
Н

?kernel
@bias
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"
_tf_keras_layer
Н

Ekernel
Fbias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+б&call_and_return_all_conditional_losses
в__call__"
_tf_keras_layer
Н

Kkernel
Lbias
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
+г&call_and_return_all_conditional_losses
д__call__"
_tf_keras_layer
Ї
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layer
Н

Ukernel
Vbias
Wtrainable_variables
Xregularization_losses
Y	variables
Z	keras_api
+з&call_and_return_all_conditional_losses
и__call__"
_tf_keras_layer

[layer-0
\layer_with_weights-0
\layer-1
]layer_with_weights-1
]layer-2
^layer_with_weights-2
^layer-3
_layer_with_weights-3
_layer-4
`layer_with_weights-4
`layer-5
alayer_with_weights-5
alayer-6
blayer_with_weights-6
blayer-7
clayer-8
dlayer-9
elayer-10
flayer-11
glayer-12
hlayer-13
itrainable_variables
jregularization_losses
k	variables
l	keras_api
+й&call_and_return_all_conditional_losses
к__call__"
_tf_keras_network

0
1
2
3
+4
,5
16
27
?8
@9
E10
F11
K12
L13
U14
V15
m16
n17
o18
p19
q20
r21
s22
t23
u24
v25
w26
x27
y28
z29"
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
2
3
+4
,5
16
27
?8
@9
E10
F11
K12
L13
U14
V15
m16
n17
o18
p19
q20
r21
s22
t23
u24
v25
w26
x27
y28
z29"
trackable_list_wrapper
Ю
trainable_variables
regularization_losses

{layers
	variables
|layer_regularization_losses
}metrics
~layer_metrics
non_trainable_variables
О__call__
М_default_save_signature
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
-
лserving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Е
trainable_variables
regularization_losses
layers
 layer_regularization_losses
	variables
metrics
layer_metrics
non_trainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_2/kernel
:2conv2d_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Е
trainable_variables
 regularization_losses
layers
 layer_regularization_losses
!	variables
metrics
layer_metrics
non_trainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
#trainable_variables
$regularization_losses
layers
 layer_regularization_losses
%	variables
metrics
layer_metrics
non_trainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
'trainable_variables
(regularization_losses
layers
 layer_regularization_losses
)	variables
metrics
layer_metrics
non_trainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
):'	2conv2d_1/kernel
:2conv2d_1/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
Е
-trainable_variables
.regularization_losses
layers
 layer_regularization_losses
/	variables
metrics
layer_metrics
non_trainable_variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_3/kernel
:2conv2d_3/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
Е
3trainable_variables
4regularization_losses
layers
 layer_regularization_losses
5	variables
metrics
layer_metrics
non_trainable_variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
7trainable_variables
8regularization_losses
layers
 layer_regularization_losses
9	variables
 metrics
Ёlayer_metrics
Ђnon_trainable_variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
;trainable_variables
<regularization_losses
Ѓlayers
 Єlayer_regularization_losses
=	variables
Ѕmetrics
Іlayer_metrics
Їnon_trainable_variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
Е
Atrainable_variables
Bregularization_losses
Јlayers
 Љlayer_regularization_losses
C	variables
Њmetrics
Ћlayer_metrics
Ќnon_trainable_variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
Е
Gtrainable_variables
Hregularization_losses
­layers
 Ўlayer_regularization_losses
I	variables
Џmetrics
Аlayer_metrics
Бnon_trainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
Е
Mtrainable_variables
Nregularization_losses
Вlayers
 Гlayer_regularization_losses
O	variables
Дmetrics
Еlayer_metrics
Жnon_trainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Qtrainable_variables
Rregularization_losses
Зlayers
 Иlayer_regularization_losses
S	variables
Йmetrics
Кlayer_metrics
Лnon_trainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
':%0 2encoder_output/kernel
!: 2encoder_output/bias
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
Е
Wtrainable_variables
Xregularization_losses
Мlayers
 Нlayer_regularization_losses
Y	variables
Оmetrics
Пlayer_metrics
Рnon_trainable_variables
и__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
7
С_init_input_shape"
_tf_keras_input_layer
С

mkernel
nbias
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
+м&call_and_return_all_conditional_losses
н__call__"
_tf_keras_layer
С

okernel
pbias
Цtrainable_variables
Чregularization_losses
Ш	variables
Щ	keras_api
+о&call_and_return_all_conditional_losses
п__call__"
_tf_keras_layer
С

qkernel
rbias
Ъtrainable_variables
Ыregularization_losses
Ь	variables
Э	keras_api
+р&call_and_return_all_conditional_losses
с__call__"
_tf_keras_layer
С

skernel
tbias
Юtrainable_variables
Яregularization_losses
а	variables
б	keras_api
+т&call_and_return_all_conditional_losses
у__call__"
_tf_keras_layer
С

ukernel
vbias
вtrainable_variables
гregularization_losses
д	variables
е	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"
_tf_keras_layer
С

wkernel
xbias
жtrainable_variables
зregularization_losses
и	variables
й	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"
_tf_keras_layer
С

ykernel
zbias
кtrainable_variables
лregularization_losses
м	variables
н	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"
_tf_keras_layer
Ћ
оtrainable_variables
пregularization_losses
р	variables
с	keras_api
+ъ&call_and_return_all_conditional_losses
ы__call__"
_tf_keras_layer
Ћ
тtrainable_variables
уregularization_losses
ф	variables
х	keras_api
+ь&call_and_return_all_conditional_losses
э__call__"
_tf_keras_layer
Ћ
цtrainable_variables
чregularization_losses
ш	variables
щ	keras_api
+ю&call_and_return_all_conditional_losses
я__call__"
_tf_keras_layer
Ћ
ъtrainable_variables
ыregularization_losses
ь	variables
э	keras_api
+№&call_and_return_all_conditional_losses
ё__call__"
_tf_keras_layer
Ћ
юtrainable_variables
яregularization_losses
№	variables
ё	keras_api
+ђ&call_and_return_all_conditional_losses
ѓ__call__"
_tf_keras_layer
Ћ
ђtrainable_variables
ѓregularization_losses
є	variables
ѕ	keras_api
+є&call_and_return_all_conditional_losses
ѕ__call__"
_tf_keras_layer

m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13"
trackable_list_wrapper
 "
trackable_list_wrapper

m0
n1
o2
p3
q4
r5
s6
t7
u8
v9
w10
x11
y12
z13"
trackable_list_wrapper
Е
itrainable_variables
jregularization_losses
іlayers
k	variables
 їlayer_regularization_losses
јmetrics
љlayer_metrics
њnon_trainable_variables
к__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_23/kernel
: 2dense_23/bias
!: 2dense_24/kernel
:2dense_24/bias
!: 2dense_26/kernel
:2dense_26/bias
!: 2dense_28/kernel
:2dense_28/bias
!:2dense_25/kernel
:2dense_25/bias
!:d2dense_27/kernel
:d2dense_27/bias
!:02dense_29/kernel
:02dense_29/bias

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
16"
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
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
И
Тtrainable_variables
Уregularization_losses
ћlayers
 ќlayer_regularization_losses
Ф	variables
§metrics
ўlayer_metrics
џnon_trainable_variables
н__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
И
Цtrainable_variables
Чregularization_losses
layers
 layer_regularization_losses
Ш	variables
metrics
layer_metrics
non_trainable_variables
п__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
И
Ъtrainable_variables
Ыregularization_losses
layers
 layer_regularization_losses
Ь	variables
metrics
layer_metrics
non_trainable_variables
с__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
И
Юtrainable_variables
Яregularization_losses
layers
 layer_regularization_losses
а	variables
metrics
layer_metrics
non_trainable_variables
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
И
вtrainable_variables
гregularization_losses
layers
 layer_regularization_losses
д	variables
metrics
layer_metrics
non_trainable_variables
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
И
жtrainable_variables
зregularization_losses
layers
 layer_regularization_losses
и	variables
metrics
layer_metrics
non_trainable_variables
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
И
кtrainable_variables
лregularization_losses
layers
 layer_regularization_losses
м	variables
metrics
layer_metrics
non_trainable_variables
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
оtrainable_variables
пregularization_losses
layers
 layer_regularization_losses
р	variables
 metrics
Ёlayer_metrics
Ђnon_trainable_variables
ы__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
тtrainable_variables
уregularization_losses
Ѓlayers
 Єlayer_regularization_losses
ф	variables
Ѕmetrics
Іlayer_metrics
Їnon_trainable_variables
э__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
цtrainable_variables
чregularization_losses
Јlayers
 Љlayer_regularization_losses
ш	variables
Њmetrics
Ћlayer_metrics
Ќnon_trainable_variables
я__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ъtrainable_variables
ыregularization_losses
­layers
 Ўlayer_regularization_losses
ь	variables
Џmetrics
Аlayer_metrics
Бnon_trainable_variables
ё__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
юtrainable_variables
яregularization_losses
Вlayers
 Гlayer_regularization_losses
№	variables
Дmetrics
Еlayer_metrics
Жnon_trainable_variables
ѓ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ђtrainable_variables
ѓregularization_losses
Зlayers
 Иlayer_regularization_losses
є	variables
Йmetrics
Кlayer_metrics
Лnon_trainable_variables
ѕ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object

[0
\1
]2
^3
_4
`5
a6
b7
c8
d9
e10
f11
g12
h13"
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
рBн
#__inference__wrapped_model_25052782input_1input_2input_3"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054499
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054650
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054191
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054275Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
-__inference_PASTA_CTAE_layer_call_fn_25053700
-__inference_PASTA_CTAE_layer_call_fn_25054721
-__inference_PASTA_CTAE_layer_call_fn_25054792
-__inference_PASTA_CTAE_layer_call_fn_25054107Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
D__inference_conv2d_layer_call_and_return_conditional_losses_25054803Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv2d_layer_call_fn_25054812Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv2d_2_layer_call_and_return_conditional_losses_25054823Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv2d_2_layer_call_fn_25054832Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_25054837
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_25054842Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
0__inference_max_pooling2d_layer_call_fn_25054847
0__inference_max_pooling2d_layer_call_fn_25054852Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ц2У
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25054857
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25054862Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
2__inference_max_pooling2d_1_layer_call_fn_25054867
2__inference_max_pooling2d_1_layer_call_fn_25054872Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv2d_1_layer_call_and_return_conditional_losses_25054883Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv2d_1_layer_call_fn_25054892Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_conv2d_3_layer_call_and_return_conditional_losses_25054903Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_conv2d_3_layer_call_fn_25054912Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_flatten_layer_call_and_return_conditional_losses_25054918Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_flatten_layer_call_fn_25054923Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_flatten_1_layer_call_and_return_conditional_losses_25054929Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_flatten_1_layer_call_fn_25054934Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_dense_layer_call_and_return_conditional_losses_25054945Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_layer_call_fn_25054954Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_1_layer_call_and_return_conditional_losses_25054965Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_1_layer_call_fn_25054974Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
я2ь
E__inference_dense_2_layer_call_and_return_conditional_losses_25054985Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
д2б
*__inference_dense_2_layer_call_fn_25054994Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѕ2ђ
K__inference_concatenate_2_layer_call_and_return_conditional_losses_25055002Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
к2з
0__inference_concatenate_2_layer_call_fn_25055009Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
і2ѓ
L__inference_encoder_output_layer_call_and_return_conditional_losses_25055019Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2и
1__inference_encoder_output_layer_call_fn_25055028Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
E__inference_decoder_layer_call_and_return_conditional_losses_25055114
E__inference_decoder_layer_call_and_return_conditional_losses_25055200
E__inference_decoder_layer_call_and_return_conditional_losses_25053368
E__inference_decoder_layer_call_and_return_conditional_losses_25053415Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
*__inference_decoder_layer_call_fn_25053057
*__inference_decoder_layer_call_fn_25055237
*__inference_decoder_layer_call_fn_25055274
*__inference_decoder_layer_call_fn_25053321Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
нBк
&__inference_signature_wrapper_25054348input_1input_2input_3"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_23_layer_call_and_return_conditional_losses_25055285Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_23_layer_call_fn_25055294Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_24_layer_call_and_return_conditional_losses_25055305Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_24_layer_call_fn_25055314Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_26_layer_call_and_return_conditional_losses_25055325Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_26_layer_call_fn_25055334Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_28_layer_call_and_return_conditional_losses_25055345Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_28_layer_call_fn_25055354Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_25_layer_call_and_return_conditional_losses_25055365Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_25_layer_call_fn_25055374Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_27_layer_call_and_return_conditional_losses_25055385Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_27_layer_call_fn_25055394Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_29_layer_call_and_return_conditional_losses_25055405Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_29_layer_call_fn_25055414Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_reshape_10_layer_call_and_return_conditional_losses_25055426Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_reshape_10_layer_call_fn_25055431Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_reshape_11_layer_call_and_return_conditional_losses_25055445Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_reshape_11_layer_call_fn_25055450Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_reshape_12_layer_call_and_return_conditional_losses_25055464Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_reshape_12_layer_call_fn_25055469Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ќ2љ
E__inference_softmax_layer_call_and_return_conditional_losses_25055474Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
с2о
*__inference_softmax_layer_call_fn_25055479Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ў2ћ
G__inference_softmax_1_layer_call_and_return_conditional_losses_25055484Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
у2р
,__inference_softmax_1_layer_call_fn_25055489Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ў2ћ
G__inference_softmax_2_layer_call_and_return_conditional_losses_25055494Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
у2р
,__inference_softmax_2_layer_call_fn_25055499Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054191Е12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ
|y
!
input_1џџџџџџџџџ
)&
input_2џџџџџџџџџ

)&
input_3џџџџџџџџџ
p 

 
Њ "zЂw
pm

0/0џџџџџџџџџ
%"
0/1џџџџџџџџџ

%"
0/2џџџџџџџџџ
 
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054275Е12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ
|y
!
input_1џџџџџџџџџ
)&
input_2џџџџџџџџџ

)&
input_3џџџџџџџџџ
p

 
Њ "zЂw
pm

0/0џџџџџџџџџ
%"
0/1џџџџџџџџџ

%"
0/2џџџџџџџџџ
 
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054499И12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ
|
"
inputs/0џџџџџџџџџ
*'
inputs/1џџџџџџџџџ

*'
inputs/2џџџџџџџџџ
p 

 
Њ "zЂw
pm

0/0џџџџџџџџџ
%"
0/1џџџџџџџџџ

%"
0/2џџџџџџџџџ
 
H__inference_PASTA_CTAE_layer_call_and_return_conditional_losses_25054650И12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ
|
"
inputs/0џџџџџџџџџ
*'
inputs/1џџџџџџџџџ

*'
inputs/2џџџџџџџџџ
p

 
Њ "zЂw
pm

0/0џџџџџџџџџ
%"
0/1џџџџџџџџџ

%"
0/2џџџџџџџџџ
 з
-__inference_PASTA_CTAE_layer_call_fn_25053700Ѕ12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ
|y
!
input_1џџџџџџџџџ
)&
input_2џџџџџџџџџ

)&
input_3џџџџџџџџџ
p 

 
Њ "jg

0џџџџџџџџџ
# 
1џџџџџџџџџ

# 
2џџџџџџџџџз
-__inference_PASTA_CTAE_layer_call_fn_25054107Ѕ12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ
|y
!
input_1џџџџџџџџџ
)&
input_2џџџџџџџџџ

)&
input_3џџџџџџџџџ
p

 
Њ "jg

0џџџџџџџџџ
# 
1џџџџџџџџџ

# 
2џџџџџџџџџк
-__inference_PASTA_CTAE_layer_call_fn_25054721Ј12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ
|
"
inputs/0џџџџџџџџџ
*'
inputs/1џџџџџџџџџ

*'
inputs/2џџџџџџџџџ
p 

 
Њ "jg

0џџџџџџџџџ
# 
1џџџџџџџџџ

# 
2џџџџџџџџџк
-__inference_PASTA_CTAE_layer_call_fn_25054792Ј12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ
|
"
inputs/0џџџџџџџџџ
*'
inputs/1џџџџџџџџџ

*'
inputs/2џџџџџџџџџ
p

 
Њ "jg

0џџџџџџџџџ
# 
1џџџџџџџџџ

# 
2џџџџџџџџџ
#__inference__wrapped_model_25052782й12+,?@EFKLUVmnstqropyzwxuvЂ
Ђ~
|y
!
input_1џџџџџџџџџ
)&
input_2џџџџџџџџџ

)&
input_3џџџџџџџџџ
Њ "ІЊЂ
,
decoder!
decoderџџџџџџџџџ
8
	decoder_1+(
	decoder_1џџџџџџџџџ

8
	decoder_2+(
	decoder_2џџџџџџџџџї
K__inference_concatenate_2_layer_call_and_return_conditional_losses_25055002Ї~Ђ{
tЂq
ol
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ0
 Я
0__inference_concatenate_2_layer_call_fn_25055009~Ђ{
tЂq
ol
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
"
inputs/2џџџџџџџџџ
Њ "џџџџџџџџџ0Ж
F__inference_conv2d_1_layer_call_and_return_conditional_losses_25054883l+,7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_conv2d_1_layer_call_fn_25054892_+,7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ " џџџџџџџџџЖ
F__inference_conv2d_2_layer_call_and_return_conditional_losses_25054823l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_conv2d_2_layer_call_fn_25054832_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџЖ
F__inference_conv2d_3_layer_call_and_return_conditional_losses_25054903l127Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
+__inference_conv2d_3_layer_call_fn_25054912_127Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџД
D__inference_conv2d_layer_call_and_return_conditional_losses_25054803l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ

Њ "-Ђ*
# 
0џџџџџџџџџ	
 
)__inference_conv2d_layer_call_fn_25054812_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ

Њ " џџџџџџџџџ	
E__inference_decoder_layer_call_and_return_conditional_losses_25053368Цmnstqropyzwxuv8Ђ5
.Ђ+
!
input_6џџџџџџџџџ 
p 

 
Њ "zЂw
pm

0/0џџџџџџџџџ
%"
0/1џџџџџџџџџ

%"
0/2џџџџџџџџџ
 
E__inference_decoder_layer_call_and_return_conditional_losses_25053415Цmnstqropyzwxuv8Ђ5
.Ђ+
!
input_6џџџџџџџџџ 
p

 
Њ "zЂw
pm

0/0џџџџџџџџџ
%"
0/1џџџџџџџџџ

%"
0/2џџџџџџџџџ
 
E__inference_decoder_layer_call_and_return_conditional_losses_25055114Хmnstqropyzwxuv7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p 

 
Њ "zЂw
pm

0/0џџџџџџџџџ
%"
0/1џџџџџџџџџ

%"
0/2џџџџџџџџџ
 
E__inference_decoder_layer_call_and_return_conditional_losses_25055200Хmnstqropyzwxuv7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p

 
Њ "zЂw
pm

0/0џџџџџџџџџ
%"
0/1џџџџџџџџџ

%"
0/2џџџџџџџџџ
 х
*__inference_decoder_layer_call_fn_25053057Жmnstqropyzwxuv8Ђ5
.Ђ+
!
input_6џџџџџџџџџ 
p 

 
Њ "jg

0џџџџџџџџџ
# 
1џџџџџџџџџ

# 
2џџџџџџџџџх
*__inference_decoder_layer_call_fn_25053321Жmnstqropyzwxuv8Ђ5
.Ђ+
!
input_6џџџџџџџџџ 
p

 
Њ "jg

0џџџџџџџџџ
# 
1џџџџџџџџџ

# 
2џџџџџџџџџф
*__inference_decoder_layer_call_fn_25055237Еmnstqropyzwxuv7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p 

 
Њ "jg

0џџџџџџџџџ
# 
1џџџџџџџџџ

# 
2џџџџџџџџџф
*__inference_decoder_layer_call_fn_25055274Еmnstqropyzwxuv7Ђ4
-Ђ*
 
inputsџџџџџџџџџ 
p

 
Њ "jg

0џџџџџџџџџ
# 
1џџџџџџџџџ

# 
2џџџџџџџџџЅ
E__inference_dense_1_layer_call_and_return_conditional_losses_25054965\EF/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_1_layer_call_fn_25054974OEF/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dense_23_layer_call_and_return_conditional_losses_25055285\mn/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ 
 ~
+__inference_dense_23_layer_call_fn_25055294Omn/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ І
F__inference_dense_24_layer_call_and_return_conditional_losses_25055305\op/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_24_layer_call_fn_25055314Oop/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџІ
F__inference_dense_25_layer_call_and_return_conditional_losses_25055365\uv/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_25_layer_call_fn_25055374Ouv/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dense_26_layer_call_and_return_conditional_losses_25055325\qr/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_26_layer_call_fn_25055334Oqr/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџІ
F__inference_dense_27_layer_call_and_return_conditional_losses_25055385\wx/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџd
 ~
+__inference_dense_27_layer_call_fn_25055394Owx/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџdІ
F__inference_dense_28_layer_call_and_return_conditional_losses_25055345\st/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_28_layer_call_fn_25055354Ost/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџІ
F__inference_dense_29_layer_call_and_return_conditional_losses_25055405\yz/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ0
 ~
+__inference_dense_29_layer_call_fn_25055414Oyz/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ0Ѕ
E__inference_dense_2_layer_call_and_return_conditional_losses_25054985\KL/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_2_layer_call_fn_25054994OKL/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЃ
C__inference_dense_layer_call_and_return_conditional_losses_25054945\?@/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 {
(__inference_dense_layer_call_fn_25054954O?@/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЌ
L__inference_encoder_output_layer_call_and_return_conditional_losses_25055019\UV/Ђ,
%Ђ"
 
inputsџџџџџџџџџ0
Њ "%Ђ"

0џџџџџџџџџ 
 
1__inference_encoder_output_layer_call_fn_25055028OUV/Ђ,
%Ђ"
 
inputsџџџџџџџџџ0
Њ "џџџџџџџџџ Ћ
G__inference_flatten_1_layer_call_and_return_conditional_losses_25054929`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
,__inference_flatten_1_layer_call_fn_25054934S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџЉ
E__inference_flatten_layer_call_and_return_conditional_losses_25054918`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
*__inference_flatten_layer_call_fn_25054923S7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "џџџџџџџџџ№
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25054857RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Й
M__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_25054862h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 Ш
2__inference_max_pooling2d_1_layer_call_fn_25054867RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2__inference_max_pooling2d_1_layer_call_fn_25054872[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџю
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_25054837RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 З
K__inference_max_pooling2d_layer_call_and_return_conditional_losses_25054842h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ "-Ђ*
# 
0џџџџџџџџџ	
 Ц
0__inference_max_pooling2d_layer_call_fn_25054847RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
0__inference_max_pooling2d_layer_call_fn_25054852[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ	
Њ " џџџџџџџџџ	Є
H__inference_reshape_10_layer_call_and_return_conditional_losses_25055426X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 |
-__inference_reshape_10_layer_call_fn_25055431K/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЌ
H__inference_reshape_11_layer_call_and_return_conditional_losses_25055445`/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ "-Ђ*
# 
0џџџџџџџџџ

 
-__inference_reshape_11_layer_call_fn_25055450S/Ђ,
%Ђ"
 
inputsџџџџџџџџџd
Њ " џџџџџџџџџ
Ќ
H__inference_reshape_12_layer_call_and_return_conditional_losses_25055464`/Ђ,
%Ђ"
 
inputsџџџџџџџџџ0
Њ "-Ђ*
# 
0џџџџџџџџџ
 
-__inference_reshape_12_layer_call_fn_25055469S/Ђ,
%Ђ"
 
inputsџџџџџџџџџ0
Њ " џџџџџџџџџЁ
&__inference_signature_wrapper_25054348і12+,?@EFKLUVmnstqropyzwxuvЊЂІ
Ђ 
Њ
,
input_1!
input_1џџџџџџџџџ
4
input_2)&
input_2џџџџџџџџџ

4
input_3)&
input_3џџџџџџџџџ"ІЊЂ
,
decoder!
decoderџџџџџџџџџ
8
	decoder_1+(
	decoder_1џџџџџџџџџ

8
	decoder_2+(
	decoder_2џџџџџџџџџЗ
G__inference_softmax_1_layer_call_and_return_conditional_losses_25055484l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ


 
Њ "-Ђ*
# 
0џџџџџџџџџ

 
,__inference_softmax_1_layer_call_fn_25055489_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ


 
Њ " џџџџџџџџџ
З
G__inference_softmax_2_layer_call_and_return_conditional_losses_25055494l;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ

 
Њ "-Ђ*
# 
0џџџџџџџџџ
 
,__inference_softmax_2_layer_call_fn_25055499_;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ

 
Њ " џџџџџџџџџЅ
E__inference_softmax_layer_call_and_return_conditional_losses_25055474\3Ђ0
)Ђ&
 
inputsџџџџџџџџџ

 
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_softmax_layer_call_fn_25055479O3Ђ0
)Ђ&
 
inputsџџџџџџџџџ

 
Њ "џџџџџџџџџ