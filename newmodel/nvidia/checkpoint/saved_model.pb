­
®ÿ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
ú
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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

ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68àá
b
gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namegamma
[
gamma/Read/ReadVariableOpReadVariableOpgamma*
_output_shapes
:*
dtype0
`
betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebeta
Y
beta/Read/ReadVariableOpReadVariableOpbeta*
_output_shapes
:*
dtype0
n
moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemoving_mean
g
moving_mean/Read/ReadVariableOpReadVariableOpmoving_mean*
_output_shapes
:*
dtype0
v
moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namemoving_variance
o
#moving_variance/Read/ReadVariableOpReadVariableOpmoving_variance*
_output_shapes
:*
dtype0
p
kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namekernel
i
kernel/Read/ReadVariableOpReadVariableOpkernel*&
_output_shapes
:*
dtype0
`
biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias
Y
bias/Read/ReadVariableOpReadVariableOpbias*
_output_shapes
:*
dtype0
t
kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_name
kernel_1
m
kernel_1/Read/ReadVariableOpReadVariableOpkernel_1*&
_output_shapes
:$*
dtype0
d
bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_namebias_1
]
bias_1/Read/ReadVariableOpReadVariableOpbias_1*
_output_shapes
:$*
dtype0
t
kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:$0*
shared_name
kernel_2
m
kernel_2/Read/ReadVariableOpReadVariableOpkernel_2*&
_output_shapes
:$0*
dtype0
d
bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namebias_2
]
bias_2/Read/ReadVariableOpReadVariableOpbias_2*
_output_shapes
:0*
dtype0
f
gamma_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name	gamma_1
_
gamma_1/Read/ReadVariableOpReadVariableOpgamma_1*
_output_shapes
:0*
dtype0
d
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namebeta_1
]
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
:0*
dtype0
r
moving_mean_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_namemoving_mean_1
k
!moving_mean_1/Read/ReadVariableOpReadVariableOpmoving_mean_1*
_output_shapes
:0*
dtype0
z
moving_variance_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_namemoving_variance_1
s
%moving_variance_1/Read/ReadVariableOpReadVariableOpmoving_variance_1*
_output_shapes
:0*
dtype0
t
kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:0@*
shared_name
kernel_3
m
kernel_3/Read/ReadVariableOpReadVariableOpkernel_3*&
_output_shapes
:0@*
dtype0
d
bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namebias_3
]
bias_3/Read/ReadVariableOpReadVariableOpbias_3*
_output_shapes
:@*
dtype0
t
kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_name
kernel_4
m
kernel_4/Read/ReadVariableOpReadVariableOpkernel_4*&
_output_shapes
:@@*
dtype0
d
bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namebias_4
]
bias_4/Read/ReadVariableOpReadVariableOpbias_4*
_output_shapes
:@*
dtype0
n
kernel_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
kernel_5
g
kernel_5/Read/ReadVariableOpReadVariableOpkernel_5* 
_output_shapes
:
*
dtype0
e
bias_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_5
^
bias_5/Read/ReadVariableOpReadVariableOpbias_5*
_output_shapes	
:*
dtype0
m
kernel_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_name
kernel_6
f
kernel_6/Read/ReadVariableOpReadVariableOpkernel_6*
_output_shapes
:	@*
dtype0
d
bias_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namebias_6
]
bias_6/Read/ReadVariableOpReadVariableOpbias_6*
_output_shapes
:@*
dtype0
l
kernel_7VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name
kernel_7
e
kernel_7/Read/ReadVariableOpReadVariableOpkernel_7*
_output_shapes

:@*
dtype0
d
bias_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_7
]
bias_7/Read/ReadVariableOpReadVariableOpbias_7*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

random_contrast/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*)
shared_namerandom_contrast/StateVar

,random_contrast/StateVar/Read/ReadVariableOpReadVariableOprandom_contrast/StateVar*
_output_shapes
:*
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
p
Adam/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/gamma/m
i
 Adam/gamma/m/Read/ReadVariableOpReadVariableOpAdam/gamma/m*
_output_shapes
:*
dtype0
n
Adam/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/beta/m
g
Adam/beta/m/Read/ReadVariableOpReadVariableOpAdam/beta/m*
_output_shapes
:*
dtype0
~
Adam/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/kernel/m
w
!Adam/kernel/m/Read/ReadVariableOpReadVariableOpAdam/kernel/m*&
_output_shapes
:*
dtype0
n
Adam/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m
g
Adam/bias/m/Read/ReadVariableOpReadVariableOpAdam/bias/m*
_output_shapes
:*
dtype0

Adam/kernel/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:$* 
shared_nameAdam/kernel/m_1
{
#Adam/kernel/m_1/Read/ReadVariableOpReadVariableOpAdam/kernel/m_1*&
_output_shapes
:$*
dtype0
r
Adam/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameAdam/bias/m_1
k
!Adam/bias/m_1/Read/ReadVariableOpReadVariableOpAdam/bias/m_1*
_output_shapes
:$*
dtype0

Adam/kernel/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:$0* 
shared_nameAdam/kernel/m_2
{
#Adam/kernel/m_2/Read/ReadVariableOpReadVariableOpAdam/kernel/m_2*&
_output_shapes
:$0*
dtype0
r
Adam/bias/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameAdam/bias/m_2
k
!Adam/bias/m_2/Read/ReadVariableOpReadVariableOpAdam/bias/m_2*
_output_shapes
:0*
dtype0
t
Adam/gamma/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameAdam/gamma/m_1
m
"Adam/gamma/m_1/Read/ReadVariableOpReadVariableOpAdam/gamma/m_1*
_output_shapes
:0*
dtype0
r
Adam/beta/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameAdam/beta/m_1
k
!Adam/beta/m_1/Read/ReadVariableOpReadVariableOpAdam/beta/m_1*
_output_shapes
:0*
dtype0

Adam/kernel/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:0@* 
shared_nameAdam/kernel/m_3
{
#Adam/kernel/m_3/Read/ReadVariableOpReadVariableOpAdam/kernel/m_3*&
_output_shapes
:0@*
dtype0
r
Adam/bias/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/bias/m_3
k
!Adam/bias/m_3/Read/ReadVariableOpReadVariableOpAdam/bias/m_3*
_output_shapes
:@*
dtype0

Adam/kernel/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameAdam/kernel/m_4
{
#Adam/kernel/m_4/Read/ReadVariableOpReadVariableOpAdam/kernel/m_4*&
_output_shapes
:@@*
dtype0
r
Adam/bias/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/bias/m_4
k
!Adam/bias/m_4/Read/ReadVariableOpReadVariableOpAdam/bias/m_4*
_output_shapes
:@*
dtype0
|
Adam/kernel/m_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/kernel/m_5
u
#Adam/kernel/m_5/Read/ReadVariableOpReadVariableOpAdam/kernel/m_5* 
_output_shapes
:
*
dtype0
s
Adam/bias/m_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m_5
l
!Adam/bias/m_5/Read/ReadVariableOpReadVariableOpAdam/bias/m_5*
_output_shapes	
:*
dtype0
{
Adam/kernel/m_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_nameAdam/kernel/m_6
t
#Adam/kernel/m_6/Read/ReadVariableOpReadVariableOpAdam/kernel/m_6*
_output_shapes
:	@*
dtype0
r
Adam/bias/m_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/bias/m_6
k
!Adam/bias/m_6/Read/ReadVariableOpReadVariableOpAdam/bias/m_6*
_output_shapes
:@*
dtype0
z
Adam/kernel/m_7VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_nameAdam/kernel/m_7
s
#Adam/kernel/m_7/Read/ReadVariableOpReadVariableOpAdam/kernel/m_7*
_output_shapes

:@*
dtype0
r
Adam/bias/m_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m_7
k
!Adam/bias/m_7/Read/ReadVariableOpReadVariableOpAdam/bias/m_7*
_output_shapes
:*
dtype0
p
Adam/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/gamma/v
i
 Adam/gamma/v/Read/ReadVariableOpReadVariableOpAdam/gamma/v*
_output_shapes
:*
dtype0
n
Adam/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/beta/v
g
Adam/beta/v/Read/ReadVariableOpReadVariableOpAdam/beta/v*
_output_shapes
:*
dtype0
~
Adam/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/kernel/v
w
!Adam/kernel/v/Read/ReadVariableOpReadVariableOpAdam/kernel/v*&
_output_shapes
:*
dtype0
n
Adam/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v
g
Adam/bias/v/Read/ReadVariableOpReadVariableOpAdam/bias/v*
_output_shapes
:*
dtype0

Adam/kernel/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:$* 
shared_nameAdam/kernel/v_1
{
#Adam/kernel/v_1/Read/ReadVariableOpReadVariableOpAdam/kernel/v_1*&
_output_shapes
:$*
dtype0
r
Adam/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:$*
shared_nameAdam/bias/v_1
k
!Adam/bias/v_1/Read/ReadVariableOpReadVariableOpAdam/bias/v_1*
_output_shapes
:$*
dtype0

Adam/kernel/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:$0* 
shared_nameAdam/kernel/v_2
{
#Adam/kernel/v_2/Read/ReadVariableOpReadVariableOpAdam/kernel/v_2*&
_output_shapes
:$0*
dtype0
r
Adam/bias/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameAdam/bias/v_2
k
!Adam/bias/v_2/Read/ReadVariableOpReadVariableOpAdam/bias/v_2*
_output_shapes
:0*
dtype0
t
Adam/gamma/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameAdam/gamma/v_1
m
"Adam/gamma/v_1/Read/ReadVariableOpReadVariableOpAdam/gamma/v_1*
_output_shapes
:0*
dtype0
r
Adam/beta/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameAdam/beta/v_1
k
!Adam/beta/v_1/Read/ReadVariableOpReadVariableOpAdam/beta/v_1*
_output_shapes
:0*
dtype0

Adam/kernel/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:0@* 
shared_nameAdam/kernel/v_3
{
#Adam/kernel/v_3/Read/ReadVariableOpReadVariableOpAdam/kernel/v_3*&
_output_shapes
:0@*
dtype0
r
Adam/bias/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/bias/v_3
k
!Adam/bias/v_3/Read/ReadVariableOpReadVariableOpAdam/bias/v_3*
_output_shapes
:@*
dtype0

Adam/kernel/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameAdam/kernel/v_4
{
#Adam/kernel/v_4/Read/ReadVariableOpReadVariableOpAdam/kernel/v_4*&
_output_shapes
:@@*
dtype0
r
Adam/bias/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/bias/v_4
k
!Adam/bias/v_4/Read/ReadVariableOpReadVariableOpAdam/bias/v_4*
_output_shapes
:@*
dtype0
|
Adam/kernel/v_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_nameAdam/kernel/v_5
u
#Adam/kernel/v_5/Read/ReadVariableOpReadVariableOpAdam/kernel/v_5* 
_output_shapes
:
*
dtype0
s
Adam/bias/v_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v_5
l
!Adam/bias/v_5/Read/ReadVariableOpReadVariableOpAdam/bias/v_5*
_output_shapes	
:*
dtype0
{
Adam/kernel/v_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_nameAdam/kernel/v_6
t
#Adam/kernel/v_6/Read/ReadVariableOpReadVariableOpAdam/kernel/v_6*
_output_shapes
:	@*
dtype0
r
Adam/bias/v_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameAdam/bias/v_6
k
!Adam/bias/v_6/Read/ReadVariableOpReadVariableOpAdam/bias/v_6*
_output_shapes
:@*
dtype0
z
Adam/kernel/v_7VarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_nameAdam/kernel/v_7
s
#Adam/kernel/v_7/Read/ReadVariableOpReadVariableOpAdam/kernel/v_7*
_output_shapes

:@*
dtype0
r
Adam/bias/v_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v_7
k
!Adam/bias/v_7/Read/ReadVariableOpReadVariableOpAdam/bias/v_7*
_output_shapes
:*
dtype0

NoOpNoOp
þµ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¸µ
value­µB©µ B¡µ
»
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer-16
layer-17
layer_with_weights-7
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer-22
layer_with_weights-9
layer-23
	optimizer

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature*
'
##_self_saveable_object_factories* 
³
#$_self_saveable_object_factories
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses* 
Ì
#+_self_saveable_object_factories
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses*
ú
3axis
	4gamma
5beta
6moving_mean
7moving_variance
#8_self_saveable_object_factories
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
Ë

?kernel
@bias
#A_self_saveable_object_factories
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
³
#H_self_saveable_object_factories
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
Ë

Okernel
Pbias
#Q_self_saveable_object_factories
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
³
#X_self_saveable_object_factories
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
Ë

_kernel
`bias
#a_self_saveable_object_factories
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
³
#h_self_saveable_object_factories
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
³
#o_self_saveable_object_factories
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
ü
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
#{_self_saveable_object_factories
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ô
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
º
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ô
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
º
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses* 
º
$¢_self_saveable_object_factories
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses* 
º
$©_self_saveable_object_factories
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses* 
Ô
°kernel
	±bias
$²_self_saveable_object_factories
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
·__call__
+¸&call_and_return_all_conditional_losses*
º
$¹_self_saveable_object_factories
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses* 
Ò
$À_self_saveable_object_factories
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å_random_generator
Æ__call__
+Ç&call_and_return_all_conditional_losses* 
Ô
Èkernel
	Ébias
$Ê_self_saveable_object_factories
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses*
º
$Ñ_self_saveable_object_factories
Ò	variables
Ótrainable_variables
Ôregularization_losses
Õ	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses* 
Ô
Økernel
	Ùbias
$Ú_self_saveable_object_factories
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses*
í
	áiter
âbeta_1
ãbeta_2

ädecay
ålearning_rate4më5mì?mí@mîOmïPmð_mñ`mòwmóxmô	mõ	mö	m÷	mø	°mù	±mú	Èmû	Émü	Ømý	Ùmþ4vÿ5v?v@vOvPv_v`vwvxv	v	v	v	v	°v	±v	Èv	Év	Øv	Ùv*

æserving_default* 
* 
Ä
40
51
62
73
?4
@5
O6
P7
_8
`9
w10
x11
y12
z13
14
15
16
17
°18
±19
È20
É21
Ø22
Ù23*
¤
40
51
?2
@3
O4
P5
_6
`7
w8
x9
10
11
12
13
°14
±15
È16
É17
Ø18
Ù19*
* 
µ
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 

ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

ö
_generator*
* 
* 
* 
TN
VARIABLE_VALUEgamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbeta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmoving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEmoving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
40
51
62
73*

40
51*
* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
VP
VARIABLE_VALUEkernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
@1*

?0
@1*
* 

ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEkernel_16layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_14layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

O0
P1*

O0
P1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEkernel_26layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_24layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

_0
`1*

_0
`1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 
* 
* 
* 
VP
VARIABLE_VALUEgamma_15layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbeta_14layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEmoving_mean_1;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEmoving_variance_1?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
w0
x1
y2
z3*

w0
x1*
* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
XR
VARIABLE_VALUEkernel_36layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_34layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEkernel_46layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_44layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
* 

®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEkernel_56layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_54layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

°0
±1*

°0
±1*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
³	variables
´trainable_variables
µregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 

Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses* 
* 
* 
* 
XR
VARIABLE_VALUEkernel_66layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_64layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

È0
É1*

È0
É1*
* 

Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
Ò	variables
Ótrainable_variables
Ôregularization_losses
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEkernel_76layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_74layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ø0
Ù1*

Ø0
Ù1*
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
60
71
y2
z3*
º
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
18
19
20
21
22
23*

à0
á1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

â
_state_var*

60
71*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

y0
z1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

ãtotal

äcount
å	variables
æ	keras_api*
<

çtotal

ècount
é	variables
ê	keras_api*
|v
VARIABLE_VALUErandom_contrast/StateVarJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ã0
ä1*

å	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

ç0
è1*

é	variables*
wq
VARIABLE_VALUEAdam/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/m_1Rlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/m_1Player_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/m_2Rlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/m_2Player_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gamma/m_1Qlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/beta/m_1Player_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/m_3Rlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/m_3Player_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/m_4Rlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/m_4Player_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/m_5Rlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/m_5Player_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/m_6Rlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/m_6Player_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/m_7Rlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/m_7Player_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/v_1Rlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/v_1Player_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/v_2Rlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/v_2Player_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/gamma/v_1Qlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/beta/v_1Player_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/v_3Rlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/v_3Player_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/v_4Rlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/v_4Player_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/v_5Rlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/v_5Player_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/v_6Rlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/v_6Player_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/kernel/v_7Rlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/bias/v_7Player_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿÀð
Ò
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1gammabetamoving_meanmoving_variancekernelbiaskernel_1bias_1kernel_2bias_2gamma_1beta_1moving_mean_1moving_variance_1kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_7bias_7*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *.
f)R'
%__inference_signature_wrapper_1443018
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamegamma/Read/ReadVariableOpbeta/Read/ReadVariableOpmoving_mean/Read/ReadVariableOp#moving_variance/Read/ReadVariableOpkernel/Read/ReadVariableOpbias/Read/ReadVariableOpkernel_1/Read/ReadVariableOpbias_1/Read/ReadVariableOpkernel_2/Read/ReadVariableOpbias_2/Read/ReadVariableOpgamma_1/Read/ReadVariableOpbeta_1/Read/ReadVariableOp!moving_mean_1/Read/ReadVariableOp%moving_variance_1/Read/ReadVariableOpkernel_3/Read/ReadVariableOpbias_3/Read/ReadVariableOpkernel_4/Read/ReadVariableOpbias_4/Read/ReadVariableOpkernel_5/Read/ReadVariableOpbias_5/Read/ReadVariableOpkernel_6/Read/ReadVariableOpbias_6/Read/ReadVariableOpkernel_7/Read/ReadVariableOpbias_7/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp,random_contrast/StateVar/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp Adam/gamma/m/Read/ReadVariableOpAdam/beta/m/Read/ReadVariableOp!Adam/kernel/m/Read/ReadVariableOpAdam/bias/m/Read/ReadVariableOp#Adam/kernel/m_1/Read/ReadVariableOp!Adam/bias/m_1/Read/ReadVariableOp#Adam/kernel/m_2/Read/ReadVariableOp!Adam/bias/m_2/Read/ReadVariableOp"Adam/gamma/m_1/Read/ReadVariableOp!Adam/beta/m_1/Read/ReadVariableOp#Adam/kernel/m_3/Read/ReadVariableOp!Adam/bias/m_3/Read/ReadVariableOp#Adam/kernel/m_4/Read/ReadVariableOp!Adam/bias/m_4/Read/ReadVariableOp#Adam/kernel/m_5/Read/ReadVariableOp!Adam/bias/m_5/Read/ReadVariableOp#Adam/kernel/m_6/Read/ReadVariableOp!Adam/bias/m_6/Read/ReadVariableOp#Adam/kernel/m_7/Read/ReadVariableOp!Adam/bias/m_7/Read/ReadVariableOp Adam/gamma/v/Read/ReadVariableOpAdam/beta/v/Read/ReadVariableOp!Adam/kernel/v/Read/ReadVariableOpAdam/bias/v/Read/ReadVariableOp#Adam/kernel/v_1/Read/ReadVariableOp!Adam/bias/v_1/Read/ReadVariableOp#Adam/kernel/v_2/Read/ReadVariableOp!Adam/bias/v_2/Read/ReadVariableOp"Adam/gamma/v_1/Read/ReadVariableOp!Adam/beta/v_1/Read/ReadVariableOp#Adam/kernel/v_3/Read/ReadVariableOp!Adam/bias/v_3/Read/ReadVariableOp#Adam/kernel/v_4/Read/ReadVariableOp!Adam/bias/v_4/Read/ReadVariableOp#Adam/kernel/v_5/Read/ReadVariableOp!Adam/bias/v_5/Read/ReadVariableOp#Adam/kernel/v_6/Read/ReadVariableOp!Adam/bias/v_6/Read/ReadVariableOp#Adam/kernel/v_7/Read/ReadVariableOp!Adam/bias/v_7/Read/ReadVariableOpConst*W
TinP
N2L		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *)
f$R"
 __inference__traced_save_1444001
»

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegammabetamoving_meanmoving_variancekernelbiaskernel_1bias_1kernel_2bias_2gamma_1beta_1moving_mean_1moving_variance_1kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_7bias_7	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_raterandom_contrast/StateVartotalcounttotal_1count_1Adam/gamma/mAdam/beta/mAdam/kernel/mAdam/bias/mAdam/kernel/m_1Adam/bias/m_1Adam/kernel/m_2Adam/bias/m_2Adam/gamma/m_1Adam/beta/m_1Adam/kernel/m_3Adam/bias/m_3Adam/kernel/m_4Adam/bias/m_4Adam/kernel/m_5Adam/bias/m_5Adam/kernel/m_6Adam/bias/m_6Adam/kernel/m_7Adam/bias/m_7Adam/gamma/vAdam/beta/vAdam/kernel/vAdam/bias/vAdam/kernel/v_1Adam/bias/v_1Adam/kernel/v_2Adam/bias/v_2Adam/gamma/v_1Adam/beta/v_1Adam/kernel/v_3Adam/bias/v_3Adam/kernel/v_4Adam/bias/v_4Adam/kernel/v_5Adam/bias/v_5Adam/kernel/v_6Adam/bias/v_6Adam/kernel/v_7Adam/bias/v_7*V
TinO
M2K*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *,
f'R%
#__inference__traced_restore_1444233µ
ø

P__inference_batch_normalization_layer_call_and_return_conditional_losses_1441329

inputs
cond_input_0:
cond_input_1:
cond_input_2:
cond_input_3:
identity¢AssignNewValue¢AssignNewValue_1¢cond;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskK
	Greater/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterGreaterstrided_slice:output:0Greater/y:output:0*
T0*
_output_shapes
: º
condIfGreater:z:0cond_input_0cond_input_1cond_input_2cond_input_3inputs*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*M
_output_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::*&
_read_only_resource_inputs
*%
else_branchR
cond_false_1441287*L
output_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::*$
then_branchR
cond_true_1441286t
cond/IdentityIdentitycond:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿO
cond/Identity_1Identitycond:output:1*
T0*
_output_shapes
:O
cond/Identity_2Identitycond:output:2*
T0*
_output_shapes
:t
AssignNewValueAssignVariableOpcond_input_2cond/Identity_1:output:0^cond*
_output_shapes
 *
dtype0v
AssignNewValue_1AssignVariableOpcond_input_3cond/Identity_2:output:0^cond*
_output_shapes
 *
dtype0
IdentityIdentitycond/Identity:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
NoOpNoOp^AssignNewValue^AssignNewValue_1^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12
condcond:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1443600

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
E
)__inference_flatten_layer_call_fn_1443605

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1441630a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¯

`
D__inference_flatten_layer_call_and_return_conditional_losses_1442162

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿu
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
â
÷)
#__inference__traced_restore_1444233
file_prefix$
assignvariableop_gamma:%
assignvariableop_1_beta:,
assignvariableop_2_moving_mean:0
"assignvariableop_3_moving_variance:3
assignvariableop_4_kernel:%
assignvariableop_5_bias:5
assignvariableop_6_kernel_1:$'
assignvariableop_7_bias_1:$5
assignvariableop_8_kernel_2:$0'
assignvariableop_9_bias_2:0)
assignvariableop_10_gamma_1:0(
assignvariableop_11_beta_1:0/
!assignvariableop_12_moving_mean_1:03
%assignvariableop_13_moving_variance_1:06
assignvariableop_14_kernel_3:0@(
assignvariableop_15_bias_3:@6
assignvariableop_16_kernel_4:@@(
assignvariableop_17_bias_4:@0
assignvariableop_18_kernel_5:
)
assignvariableop_19_bias_5:	/
assignvariableop_20_kernel_6:	@(
assignvariableop_21_bias_6:@.
assignvariableop_22_kernel_7:@(
assignvariableop_23_bias_7:'
assignvariableop_24_adam_iter:	 )
assignvariableop_25_adam_beta_1: )
assignvariableop_26_adam_beta_2: (
assignvariableop_27_adam_decay: 0
&assignvariableop_28_adam_learning_rate: :
,assignvariableop_29_random_contrast_statevar:	#
assignvariableop_30_total: #
assignvariableop_31_count: %
assignvariableop_32_total_1: %
assignvariableop_33_count_1: .
 assignvariableop_34_adam_gamma_m:-
assignvariableop_35_adam_beta_m:;
!assignvariableop_36_adam_kernel_m:-
assignvariableop_37_adam_bias_m:=
#assignvariableop_38_adam_kernel_m_1:$/
!assignvariableop_39_adam_bias_m_1:$=
#assignvariableop_40_adam_kernel_m_2:$0/
!assignvariableop_41_adam_bias_m_2:00
"assignvariableop_42_adam_gamma_m_1:0/
!assignvariableop_43_adam_beta_m_1:0=
#assignvariableop_44_adam_kernel_m_3:0@/
!assignvariableop_45_adam_bias_m_3:@=
#assignvariableop_46_adam_kernel_m_4:@@/
!assignvariableop_47_adam_bias_m_4:@7
#assignvariableop_48_adam_kernel_m_5:
0
!assignvariableop_49_adam_bias_m_5:	6
#assignvariableop_50_adam_kernel_m_6:	@/
!assignvariableop_51_adam_bias_m_6:@5
#assignvariableop_52_adam_kernel_m_7:@/
!assignvariableop_53_adam_bias_m_7:.
 assignvariableop_54_adam_gamma_v:-
assignvariableop_55_adam_beta_v:;
!assignvariableop_56_adam_kernel_v:-
assignvariableop_57_adam_bias_v:=
#assignvariableop_58_adam_kernel_v_1:$/
!assignvariableop_59_adam_bias_v_1:$=
#assignvariableop_60_adam_kernel_v_2:$0/
!assignvariableop_61_adam_bias_v_2:00
"assignvariableop_62_adam_gamma_v_1:0/
!assignvariableop_63_adam_beta_v_1:0=
#assignvariableop_64_adam_kernel_v_3:0@/
!assignvariableop_65_adam_bias_v_3:@=
#assignvariableop_66_adam_kernel_v_4:@@/
!assignvariableop_67_adam_bias_v_4:@7
#assignvariableop_68_adam_kernel_v_5:
0
!assignvariableop_69_adam_bias_v_5:	6
#assignvariableop_70_adam_kernel_v_6:	@/
!assignvariableop_71_adam_bias_v_6:@5
#assignvariableop_72_adam_kernel_v_7:@/
!assignvariableop_73_adam_bias_v_7:
identity_75¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_8¢AssignVariableOp_9Ð)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*ö(
valueì(Bé(KB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_kernel_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_bias_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_kernel_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_bias_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_gamma_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp!assignvariableop_12_moving_mean_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_moving_variance_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_kernel_3Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_bias_3Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_kernel_4Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_bias_4Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_kernel_5Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_bias_5Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_kernel_6Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_bias_6Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_kernel_7Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_bias_7Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_29AssignVariableOp,assignvariableop_29_random_contrast_statevarIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp assignvariableop_34_adam_gamma_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_beta_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp!assignvariableop_36_adam_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp#assignvariableop_38_adam_kernel_m_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp!assignvariableop_39_adam_bias_m_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp#assignvariableop_40_adam_kernel_m_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp!assignvariableop_41_adam_bias_m_2Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp"assignvariableop_42_adam_gamma_m_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp!assignvariableop_43_adam_beta_m_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp#assignvariableop_44_adam_kernel_m_3Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp!assignvariableop_45_adam_bias_m_3Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp#assignvariableop_46_adam_kernel_m_4Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp!assignvariableop_47_adam_bias_m_4Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp#assignvariableop_48_adam_kernel_m_5Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp!assignvariableop_49_adam_bias_m_5Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp#assignvariableop_50_adam_kernel_m_6Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp!assignvariableop_51_adam_bias_m_6Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp#assignvariableop_52_adam_kernel_m_7Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp!assignvariableop_53_adam_bias_m_7Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp assignvariableop_54_adam_gamma_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp!assignvariableop_56_adam_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp#assignvariableop_58_adam_kernel_v_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp!assignvariableop_59_adam_bias_v_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp#assignvariableop_60_adam_kernel_v_2Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp!assignvariableop_61_adam_bias_v_2Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp"assignvariableop_62_adam_gamma_v_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp!assignvariableop_63_adam_beta_v_1Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp#assignvariableop_64_adam_kernel_v_3Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp!assignvariableop_65_adam_bias_v_3Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp#assignvariableop_66_adam_kernel_v_4Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp!assignvariableop_67_adam_bias_v_4Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp#assignvariableop_68_adam_kernel_v_5Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp!assignvariableop_69_adam_bias_v_5Identity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp#assignvariableop_70_adam_kernel_v_6Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp!assignvariableop_71_adam_bias_v_6Identity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp#assignvariableop_72_adam_kernel_v_7Identity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp!assignvariableop_73_adam_bias_v_7Identity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 «
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_75IdentityIdentity_74:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_75Identity_75:output:0*«
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
^
î

B__inference_model_layer_call_and_return_conditional_losses_1441702

inputs)
batch_normalization_1441489:)
batch_normalization_1441491:)
batch_normalization_1441493:)
batch_normalization_1441495:(
conv2d_1441509:
conv2d_1441511:*
conv2d_1_1441532:$
conv2d_1_1441534:$*
conv2d_2_1441555:$0
conv2d_2_1441557:0+
batch_normalization_1_1441568:0+
batch_normalization_1_1441570:0+
batch_normalization_1_1441572:0+
batch_normalization_1_1441574:0*
conv2d_3_1441588:0@
conv2d_3_1441590:@*
conv2d_4_1441611:@@
conv2d_4_1441613:@!
dense_1441643:

dense_1441645:	"
dense_1_1441673:	@
dense_1_1441675:@!
dense_2_1441696:@
dense_2_1441698:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallÈ
resizing/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_resizing_layer_call_and_return_conditional_losses_1441481ñ
random_contrast/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_1441487
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(random_contrast/PartitionedCall:output:0batch_normalization_1441489batch_normalization_1441491batch_normalization_1441493batch_normalization_1441495*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1441263¦
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1441509conv2d_1441511*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1441508ë
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1441519
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_1441532conv2d_1_1441534*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1441531ñ
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1441542
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_1441555conv2d_2_1441557*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1441554ñ
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1441565ï
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1441349
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_1441568batch_normalization_1_1441570batch_normalization_1_1441572batch_normalization_1_1441574*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1441379°
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_3_1441588conv2d_3_1441590*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1441587ñ
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1441598
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_1441611conv2d_4_1441613*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1441610ñ
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1441621ó
max_pooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1441465ß
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1441630
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1441643dense_1441645*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1441642ç
activation_5/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1441653Ü
dropout/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1441660
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_1441673dense_1_1441675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1441672è
activation_6/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1441683
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_2_1441696dense_2_1441698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1441695w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs
Ú
M
1__inference_random_contrast_layer_call_fn_1443034

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_1441487j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿàà:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

¤
cond_false_1443432
cond_placeholder
cond_placeholder_1*
cond_readvariableop_resource:0,
cond_readvariableop_1_resource:0
cond_identity_inputs
cond_identity
cond_identity_1
cond_identity_2¢cond/ReadVariableOp¢cond/ReadVariableOp_1
cond/IdentityIdentitycond_identity_inputs
^cond/NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:0*
dtype0i
cond/Identity_1Identitycond/ReadVariableOp:value:0
^cond/NoOp*
T0*
_output_shapes
:0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:0*
dtype0k
cond/Identity_2Identitycond/ReadVariableOp_1:value:0
^cond/NoOp*
T0*
_output_shapes
:0y
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
í
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1441565

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
í
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1443357

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs

h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1441465

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

þ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1443342

inputs8
conv2d_readvariableop_resource:$0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1443560

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ

@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
 
_user_specified_nameinputs
ý
¢
&batch_normalization_cond_false_1442805(
$batch_normalization_cond_placeholder*
&batch_normalization_cond_placeholder_1>
0batch_normalization_cond_readvariableop_resource:@
2batch_normalization_cond_readvariableop_1_resource:N
Jbatch_normalization_cond_identity_random_contrast_adjust_contrast_identity%
!batch_normalization_cond_identity'
#batch_normalization_cond_identity_1'
#batch_normalization_cond_identity_2¢'batch_normalization/cond/ReadVariableOp¢)batch_normalization/cond/ReadVariableOp_1Õ
!batch_normalization/cond/IdentityIdentityJbatch_normalization_cond_identity_random_contrast_adjust_contrast_identity^batch_normalization/cond/NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
'batch_normalization/cond/ReadVariableOpReadVariableOp0batch_normalization_cond_readvariableop_resource*
_output_shapes
:*
dtype0¥
#batch_normalization/cond/Identity_1Identity/batch_normalization/cond/ReadVariableOp:value:0^batch_normalization/cond/NoOp*
T0*
_output_shapes
:
)batch_normalization/cond/ReadVariableOp_1ReadVariableOp2batch_normalization_cond_readvariableop_1_resource*
_output_shapes
:*
dtype0§
#batch_normalization/cond/Identity_2Identity1batch_normalization/cond/ReadVariableOp_1:value:0^batch_normalization/cond/NoOp*
T0*
_output_shapes
:µ
batch_normalization/cond/NoOpNoOp(^batch_normalization/cond/ReadVariableOp*^batch_normalization/cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "O
!batch_normalization_cond_identity*batch_normalization/cond/Identity:output:0"S
#batch_normalization_cond_identity_1,batch_normalization/cond/Identity_1:output:0"S
#batch_normalization_cond_identity_2,batch_normalization/cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿàà2R
'batch_normalization/cond/ReadVariableOp'batch_normalization/cond/ReadVariableOp2V
)batch_normalization/cond/ReadVariableOp_1)batch_normalization/cond/ReadVariableOp_1:73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
Û
b
D__inference_dropout_layer_call_and_return_conditional_losses_1441660

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º

*__inference_conv2d_1_layer_call_fn_1443264

inputs!
unknown:$
	unknown_0:$
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1442065
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
F
*__inference_resizing_layer_call_fn_1443023

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_resizing_layer_call_and_return_conditional_losses_1441481j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀð:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs
	
Ð
5__inference_batch_normalization_layer_call_fn_1443112

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1441329
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
K
/__inference_max_pooling2d_layer_call_fn_1443367

inputs
identityÝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1441349
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í
e
I__inference_activation_6_layer_call_and_return_conditional_losses_1441683

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½c
á
B__inference_model_layer_call_and_return_conditional_losses_1442193

inputs%
random_contrast_1442023:	)
batch_normalization_1442026:)
batch_normalization_1442028:)
batch_normalization_1442030:)
batch_normalization_1442032:(
conv2d_1442045:
conv2d_1442047:*
conv2d_1_1442066:$
conv2d_1_1442068:$*
conv2d_2_1442087:$0
conv2d_2_1442089:0+
batch_normalization_1_1442099:0+
batch_normalization_1_1442101:0+
batch_normalization_1_1442103:0+
batch_normalization_1_1442105:0*
conv2d_3_1442118:0@
conv2d_3_1442120:@*
conv2d_4_1442139:@@
conv2d_4_1442141:@!
dense_1442174:

dense_1442176:	"
dense_1_1442181:	@
dense_1_1442183:@!
dense_2_1442187:@
dense_2_1442189:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢'random_contrast/StatefulPartitionedCallÈ
resizing/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_resizing_layer_call_and_return_conditional_losses_1441481
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall!resizing/PartitionedCall:output:0random_contrast_1442023*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_1441953¤
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0random_contrast/StatefulPartitionedCall:output:0batch_normalization_1442026batch_normalization_1442028batch_normalization_1442030batch_normalization_1442032*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1441329¸
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1442045conv2d_1442047*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1442044ý
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1442054¯
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_1442066conv2d_1_1442068*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1442065
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1442075±
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_1442087conv2d_2_1442089*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1442086
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1442096
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1441349¦
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_1442099batch_normalization_1_1442101batch_normalization_1_1442103batch_normalization_1_1442105*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1441445Â
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_3_1442118conv2d_3_1442120*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1442117
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1442127±
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_1442139conv2d_4_1442141*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1442138
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1442148
max_pooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1441465ç
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1442162
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1442174dense_1442176*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1442173ç
activation_5/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1441653ì
dropout/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1441799
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_1442181dense_1_1442183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1441672è
activation_6/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1441683
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_2_1442187dense_2_1442189*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1441695w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall(^random_contrast/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs

þ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1442138

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¶

(__inference_conv2d_layer_call_fn_1443206

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1442044
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú

'__inference_dense_layer_call_fn_1443646

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1442173p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à	
ö
B__inference_dense_layer_call_and_return_conditional_losses_1443666

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


%__inference_signature_wrapper_1443018
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:$
	unknown_6:$#
	unknown_7:$0
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0@

unknown_14:@$

unknown_15:@@

unknown_16:@

unknown_17:


unknown_18:	

unknown_19:	@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *+
f&R$
"__inference__wrapped_model_1441236o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
!
_user_specified_name	input_1

J
.__inference_activation_2_layer_call_fn_1443352

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1442096z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ø

P__inference_batch_normalization_layer_call_and_return_conditional_losses_1443188

inputs
cond_input_0:
cond_input_1:
cond_input_2:
cond_input_3:
identity¢AssignNewValue¢AssignNewValue_1¢cond;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskK
	Greater/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterGreaterstrided_slice:output:0Greater/y:output:0*
T0*
_output_shapes
: º
condIfGreater:z:0cond_input_0cond_input_1cond_input_2cond_input_3inputs*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*M
_output_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::*&
_read_only_resource_inputs
*%
else_branchR
cond_false_1443146*L
output_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::*$
then_branchR
cond_true_1443145t
cond/IdentityIdentitycond:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿO
cond/Identity_1Identitycond:output:1*
T0*
_output_shapes
:O
cond/Identity_2Identitycond:output:2*
T0*
_output_shapes
:t
AssignNewValueAssignVariableOpcond_input_2cond/Identity_1:output:0^cond*
_output_shapes
 *
dtype0v
AssignNewValue_1AssignVariableOpcond_input_3cond/Identity_2:output:0^cond*
_output_shapes
 *
dtype0
IdentityIdentitycond/Identity:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
NoOpNoOp^AssignNewValue^AssignNewValue_1^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12
condcond:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º

*__inference_conv2d_3_layer_call_fn_1443492

inputs!
unknown:0@
	unknown_0:@
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1442117
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
µ
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1442127

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

þ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1443512

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Ù
É
cond_true_1441286*
cond_readvariableop_resource:,
cond_readvariableop_1_resource:;
-cond_fusedbatchnormv3_readvariableop_resource:=
/cond_fusedbatchnormv3_readvariableop_1_resource: 
cond_fusedbatchnormv3_inputs
cond_identity
cond_identity_1
cond_identity_2¢$cond/FusedBatchNormV3/ReadVariableOp¢&cond/FusedBatchNormV3/ReadVariableOp_1¢cond/ReadVariableOp¢cond/ReadVariableOp_1l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:*
dtype0
$cond/FusedBatchNormV3/ReadVariableOpReadVariableOp-cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
&cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
cond/FusedBatchNormV3FusedBatchNormV3cond_fusedbatchnormv3_inputscond/ReadVariableOp:value:0cond/ReadVariableOp_1:value:0,cond/FusedBatchNormV3/ReadVariableOp:value:0.cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
cond/IdentityIdentitycond/FusedBatchNormV3:y:0
^cond/NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
cond/Identity_1Identity"cond/FusedBatchNormV3:batch_mean:0
^cond/NoOp*
T0*
_output_shapes
:t
cond/Identity_2Identity&cond/FusedBatchNormV3:batch_variance:0
^cond/NoOp*
T0*
_output_shapes
:É
	cond/NoOpNoOp%^cond/FusedBatchNormV3/ReadVariableOp'^cond/FusedBatchNormV3/ReadVariableOp_1^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2L
$cond/FusedBatchNormV3/ReadVariableOp$cond/FusedBatchNormV3/ReadVariableOp2P
&cond/FusedBatchNormV3/ReadVariableOp_1&cond/FusedBatchNormV3/ReadVariableOp_12*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
	
Ò
7__inference_batch_normalization_1_layer_call_fn_1443398

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1441445
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs

ü
C__inference_conv2d_layer_call_and_return_conditional_losses_1442044

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý

P__inference_batch_normalization_layer_call_and_return_conditional_losses_1441263

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

þ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1442117

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Àc
â
B__inference_model_layer_call_and_return_conditional_losses_1442537
input_1%
random_contrast_1442464:	)
batch_normalization_1442467:)
batch_normalization_1442469:)
batch_normalization_1442471:)
batch_normalization_1442473:(
conv2d_1442476:
conv2d_1442478:*
conv2d_1_1442482:$
conv2d_1_1442484:$*
conv2d_2_1442488:$0
conv2d_2_1442490:0+
batch_normalization_1_1442495:0+
batch_normalization_1_1442497:0+
batch_normalization_1_1442499:0+
batch_normalization_1_1442501:0*
conv2d_3_1442504:0@
conv2d_3_1442506:@*
conv2d_4_1442510:@@
conv2d_4_1442512:@!
dense_1442518:

dense_1442520:	"
dense_1_1442525:	@
dense_1_1442527:@!
dense_2_1442531:@
dense_2_1442533:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢'random_contrast/StatefulPartitionedCallÉ
resizing/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_resizing_layer_call_and_return_conditional_losses_1441481
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall!resizing/PartitionedCall:output:0random_contrast_1442464*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_1441953¤
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0random_contrast/StatefulPartitionedCall:output:0batch_normalization_1442467batch_normalization_1442469batch_normalization_1442471batch_normalization_1442473*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1441329¸
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1442476conv2d_1442478*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1442044ý
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1442054¯
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_1442482conv2d_1_1442484*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1442065
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1442075±
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_1442488conv2d_2_1442490*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1442086
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1442096
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1441349¦
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_1442495batch_normalization_1_1442497batch_normalization_1_1442499batch_normalization_1_1442501*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1441445Â
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_3_1442504conv2d_3_1442506*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1442117
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1442127±
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_1442510conv2d_4_1442512*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1442138
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1442148
max_pooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1441465ç
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1442162
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1442518dense_1442520*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1442173ç
activation_5/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1441653ì
dropout/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1441799
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_1442525dense_1_1442527*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1441672è
activation_6/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1441683
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_2_1442531dense_2_1442533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1441695w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall(^random_contrast/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
!
_user_specified_name	input_1
Û
b
D__inference_dropout_layer_call_and_return_conditional_losses_1443691

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð	
ö
B__inference_dense_layer_call_and_return_conditional_losses_1443656

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ

*__inference_conv2d_4_layer_call_fn_1443541

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1441610w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ

@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
 
_user_specified_nameinputs
Ü
Ê
"__inference__wrapped_model_1441236
input_1?
1model_batch_normalization_readvariableop_resource:A
3model_batch_normalization_readvariableop_1_resource:P
Bmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource:R
Dmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:E
+model_conv2d_conv2d_readvariableop_resource::
,model_conv2d_biasadd_readvariableop_resource:G
-model_conv2d_1_conv2d_readvariableop_resource:$<
.model_conv2d_1_biasadd_readvariableop_resource:$G
-model_conv2d_2_conv2d_readvariableop_resource:$0<
.model_conv2d_2_biasadd_readvariableop_resource:0A
3model_batch_normalization_1_readvariableop_resource:0C
5model_batch_normalization_1_readvariableop_1_resource:0R
Dmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0T
Fmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0G
-model_conv2d_3_conv2d_readvariableop_resource:0@<
.model_conv2d_3_biasadd_readvariableop_resource:@G
-model_conv2d_4_conv2d_readvariableop_resource:@@<
.model_conv2d_4_biasadd_readvariableop_resource:@>
*model_dense_matmul_readvariableop_resource:
:
+model_dense_biasadd_readvariableop_resource:	?
,model_dense_1_matmul_readvariableop_resource:	@;
-model_dense_1_biasadd_readvariableop_resource:@>
,model_dense_2_matmul_readvariableop_resource:@;
-model_dense_2_biasadd_readvariableop_resource:
identity¢9model/batch_normalization/FusedBatchNormV3/ReadVariableOp¢;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢(model/batch_normalization/ReadVariableOp¢*model/batch_normalization/ReadVariableOp_1¢;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢*model/batch_normalization_1/ReadVariableOp¢,model/batch_normalization_1/ReadVariableOp_1¢#model/conv2d/BiasAdd/ReadVariableOp¢"model/conv2d/Conv2D/ReadVariableOp¢%model/conv2d_1/BiasAdd/ReadVariableOp¢$model/conv2d_1/Conv2D/ReadVariableOp¢%model/conv2d_2/BiasAdd/ReadVariableOp¢$model/conv2d_2/Conv2D/ReadVariableOp¢%model/conv2d_3/BiasAdd/ReadVariableOp¢$model/conv2d_3/Conv2D/ReadVariableOp¢%model/conv2d_4/BiasAdd/ReadVariableOp¢$model/conv2d_4/Conv2D/ReadVariableOp¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢$model/dense_1/BiasAdd/ReadVariableOp¢#model/dense_1/MatMul/ReadVariableOp¢$model/dense_2/BiasAdd/ReadVariableOp¢#model/dense_2/MatMul/ReadVariableOpk
model/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   º
$model/resizing/resize/ResizeBilinearResizeBilinearinput_1#model/resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(
model/batch_normalization/ShapeShape5model/resizing/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:w
-model/batch_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model/batch_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model/batch_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'model/batch_normalization/strided_sliceStridedSlice(model/batch_normalization/Shape:output:06model/batch_normalization/strided_slice/stack:output:08model/batch_normalization/strided_slice/stack_1:output:08model/batch_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0¸
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0¼
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0é
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV35model/resizing/resize/ResizeBilinear:resized_images:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿàà:::::*
epsilon%o:*
is_training( 
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ü
model/conv2d/Conv2DConv2D.model/batch_normalization/FusedBatchNormV3:y:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn*
paddingVALID*
strides

#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¤
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnnv
model/activation/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0Õ
model/conv2d_1/Conv2DConv2D#model/activation/Relu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$*
paddingVALID*
strides

%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0ª
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$z
model/activation_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0×
model/conv2d_2/Conv2DConv2D%model/activation_1/Relu:activations:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides

%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0ª
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0z
model/activation_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0º
model/max_pooling2d/MaxPoolMaxPool%model/activation_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
ksize
*
paddingVALID*
strides
u
!model/batch_normalization_1/ShapeShape$model/max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:y
/model/batch_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model/batch_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model/batch_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)model/batch_normalization_1/strided_sliceStridedSlice*model/batch_normalization_1/Shape:output:08model/batch_normalization_1/strided_slice/stack:output:0:model/batch_normalization_1/strided_slice/stack_1:output:0:model/batch_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype0
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype0¼
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0À
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0à
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$model/max_pooling2d/MaxPool:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0â
model/conv2d_3/Conv2DConv2D0model/batch_normalization_1/FusedBatchNormV3:y:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
paddingVALID*
strides

%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ª
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@z
model/activation_3/ReluRelumodel/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0×
model/conv2d_4/Conv2DConv2D%model/activation_3/Relu:activations:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ª
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
model/activation_4/ReluRelumodel/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
model/max_pooling2d_1/MaxPoolMaxPool%model/activation_4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
model/flatten/ReshapeReshape&model/max_pooling2d_1/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
model/activation_5/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
model/dropout/IdentityIdentity%model/activation_5/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0 
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@q
model/activation_6/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¤
model/dense_2/MatMulMatMul%model/activation_6/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentitymodel/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
NoOpNoOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : 2v
9model/batch_normalization/FusedBatchNormV3/ReadVariableOp9model/batch_normalization/FusedBatchNormV3/ReadVariableOp2z
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_12T
(model/batch_normalization/ReadVariableOp(model/batch_normalization/ReadVariableOp2X
*model/batch_normalization/ReadVariableOp_1*model/batch_normalization/ReadVariableOp_12z
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2~
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12X
*model/batch_normalization_1/ReadVariableOp*model/batch_normalization_1/ReadVariableOp2\
,model/batch_normalization_1/ReadVariableOp_1,model/batch_normalization_1/ReadVariableOp_12J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2N
%model/conv2d_3/BiasAdd/ReadVariableOp%model/conv2d_3/BiasAdd/ReadVariableOp2L
$model/conv2d_3/Conv2D/ReadVariableOp$model/conv2d_3/Conv2D/ReadVariableOp2N
%model/conv2d_4/BiasAdd/ReadVariableOp%model/conv2d_4/BiasAdd/ReadVariableOp2L
$model/conv2d_4/Conv2D/ReadVariableOp$model/conv2d_4/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
!
_user_specified_name	input_1
¬

'__inference_model_layer_call_fn_1441753
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:$
	unknown_6:$#
	unknown_7:$0
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0@

unknown_14:@$

unknown_15:@@

unknown_16:@

unknown_17:


unknown_18:	

unknown_19:	@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1441702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
!
_user_specified_name	input_1
º

*__inference_conv2d_4_layer_call_fn_1443550

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1442138
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

J
.__inference_activation_4_layer_call_fn_1443580

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1442148z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ñ
e
I__inference_activation_5_layer_call_and_return_conditional_losses_1441653

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1442096

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
í
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1443299

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ55$:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
 
_user_specified_nameinputs
¿
M
1__inference_max_pooling2d_1_layer_call_fn_1443595

inputs
identityß
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1441465
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

þ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1442065

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
J
.__inference_activation_3_layer_call_fn_1443517

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1441598h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
 
_user_specified_nameinputs
Æ
`
D__inference_flatten_layer_call_and_return_conditional_losses_1441630

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
í
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1441542

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ55$:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
 
_user_specified_nameinputs
Ì
J
.__inference_activation_4_layer_call_fn_1443575

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1441621h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È
H
,__inference_activation_layer_call_fn_1443231

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1441519h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿnn:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
 
_user_specified_nameinputs
Ë	
ö
D__inference_dense_1_layer_call_and_return_conditional_losses_1443722

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1441379

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
í
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1443527

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
 
_user_specified_nameinputs
Ñ
e
I__inference_activation_5_layer_call_and_return_conditional_losses_1443676

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
e
I__inference_activation_2_layer_call_and_return_conditional_losses_1443362

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1443332

inputs8
conv2d_readvariableop_resource:$0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ55$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
 
_user_specified_nameinputs
ú

R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1443474

inputs
cond_input_0:0
cond_input_1:0
cond_input_2:0
cond_input_3:0
identity¢AssignNewValue¢AssignNewValue_1¢cond;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskK
	Greater/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterGreaterstrided_slice:output:0Greater/y:output:0*
T0*
_output_shapes
: º
condIfGreater:z:0cond_input_0cond_input_1cond_input_2cond_input_3inputs*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*M
_output_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0*&
_read_only_resource_inputs
*%
else_branchR
cond_false_1443432*L
output_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0*$
then_branchR
cond_true_1443431t
cond/IdentityIdentitycond:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0O
cond/Identity_1Identitycond:output:1*
T0*
_output_shapes
:0O
cond/Identity_2Identitycond:output:2*
T0*
_output_shapes
:0t
AssignNewValueAssignVariableOpcond_input_2cond/Identity_1:output:0^cond*
_output_shapes
 *
dtype0v
AssignNewValue_1AssignVariableOpcond_input_3cond/Identity_2:output:0^cond*
_output_shapes
 *
dtype0
IdentityIdentitycond/Identity:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0q
NoOpNoOp^AssignNewValue^AssignNewValue_1^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12
condcond:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
µ
e
I__inference_activation_4_layer_call_and_return_conditional_losses_1442148

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤/
×
L__inference_random_contrast_layer_call_and_return_conditional_losses_1441953

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maska
stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ì
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: 
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¢
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: |
adjust_contrastAdjustContrastv2inputsstateless_random_uniform:z:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààz
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààz
IdentityIdentity!adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààq
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿàà: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

þ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1442086

inputs8
conv2d_readvariableop_resource:$0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
µ
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1443532

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ
`
D__inference_flatten_layer_call_and_return_conditional_losses_1443616

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ú	
c
D__inference_dropout_layer_call_and_return_conditional_losses_1441799

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú

R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1441445

inputs
cond_input_0:0
cond_input_1:0
cond_input_2:0
cond_input_3:0
identity¢AssignNewValue¢AssignNewValue_1¢cond;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskK
	Greater/yConst*
_output_shapes
: *
dtype0*
value	B : _
GreaterGreaterstrided_slice:output:0Greater/y:output:0*
T0*
_output_shapes
: º
condIfGreater:z:0cond_input_0cond_input_1cond_input_2cond_input_3inputs*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*M
_output_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0*&
_read_only_resource_inputs
*%
else_branchR
cond_false_1441403*L
output_shapes;
9:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0*$
then_branchR
cond_true_1441402t
cond/IdentityIdentitycond:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0O
cond/Identity_1Identitycond:output:1*
T0*
_output_shapes
:0O
cond/Identity_2Identitycond:output:2*
T0*
_output_shapes
:0t
AssignNewValueAssignVariableOpcond_input_2cond/Identity_1:output:0^cond*
_output_shapes
 *
dtype0v
AssignNewValue_1AssignVariableOpcond_input_3cond/Identity_2:output:0^cond*
_output_shapes
 *
dtype0
IdentityIdentitycond/Identity:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0q
NoOpNoOp^AssignNewValue^AssignNewValue_1^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12
condcond:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
ñ

*__inference_conv2d_2_layer_call_fn_1443313

inputs!
unknown:$0
	unknown_0:0
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1441554w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ55$: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
 
_user_specified_nameinputs
ñ

(__inference_conv2d_layer_call_fn_1443197

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1441508w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1441349

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
J
.__inference_activation_1_layer_call_fn_1443289

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1441542h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ55$:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
 
_user_specified_nameinputs
è
E
)__inference_flatten_layer_call_fn_1443610

inputs
identity½
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1442162i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤¼

B__inference_model_layer_call_and_return_conditional_losses_1442963

inputsO
Arandom_contrast_stateful_uniform_full_int_rngreadandskip_resource:	.
 batch_normalization_cond_input_0:.
 batch_normalization_cond_input_1:.
 batch_normalization_cond_input_2:.
 batch_normalization_cond_input_3:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:$6
(conv2d_1_biasadd_readvariableop_resource:$A
'conv2d_2_conv2d_readvariableop_resource:$06
(conv2d_2_biasadd_readvariableop_resource:00
"batch_normalization_1_cond_input_0:00
"batch_normalization_1_cond_input_1:00
"batch_normalization_1_cond_input_2:00
"batch_normalization_1_cond_input_3:0A
'conv2d_3_conv2d_readvariableop_resource:0@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1¢batch_normalization/cond¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1¢batch_normalization_1/cond¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOp¢8random_contrast/stateful_uniform_full_int/RngReadAndSkipe
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   ­
resizing/resize/ResizeBilinearResizeBilinearinputsresizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(y
/random_contrast/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:y
/random_contrast/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ë
.random_contrast/stateful_uniform_full_int/ProdProd8random_contrast/stateful_uniform_full_int/shape:output:08random_contrast/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: r
0random_contrast/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :¡
0random_contrast/stateful_uniform_full_int/Cast_1Cast7random_contrast/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
8random_contrast/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipArandom_contrast_stateful_uniform_full_int_rngreadandskip_resource9random_contrast/stateful_uniform_full_int/Cast/x:output:04random_contrast/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:
=random_contrast/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?random_contrast/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?random_contrast/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:©
7random_contrast/stateful_uniform_full_int/strided_sliceStridedSlice@random_contrast/stateful_uniform_full_int/RngReadAndSkip:value:0Frandom_contrast/stateful_uniform_full_int/strided_slice/stack:output:0Hrandom_contrast/stateful_uniform_full_int/strided_slice/stack_1:output:0Hrandom_contrast/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask¯
1random_contrast/stateful_uniform_full_int/BitcastBitcast@random_contrast/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
?random_contrast/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Arandom_contrast/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Arandom_contrast/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
9random_contrast/stateful_uniform_full_int/strided_slice_1StridedSlice@random_contrast/stateful_uniform_full_int/RngReadAndSkip:value:0Hrandom_contrast/stateful_uniform_full_int/strided_slice_1/stack:output:0Jrandom_contrast/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Jrandom_contrast/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:³
3random_contrast/stateful_uniform_full_int/Bitcast_1BitcastBrandom_contrast/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-random_contrast/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :á
)random_contrast/stateful_uniform_full_intStatelessRandomUniformFullIntV28random_contrast/stateful_uniform_full_int/shape:output:0<random_contrast/stateful_uniform_full_int/Bitcast_1:output:0:random_contrast/stateful_uniform_full_int/Bitcast:output:06random_contrast/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	d
random_contrast/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ¨
random_contrast/stackPack2random_contrast/stateful_uniform_full_int:output:0#random_contrast/zeros_like:output:0*
N*
T0	*
_output_shapes

:t
#random_contrast/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%random_contrast/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%random_contrast/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
random_contrast/strided_sliceStridedSlicerandom_contrast/stack:output:0,random_contrast/strided_slice/stack:output:0.random_contrast/strided_slice/stack_1:output:0.random_contrast/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maskq
.random_contrast/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB q
,random_contrast/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?q
,random_contrast/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?¯
Erandom_contrast/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter&random_contrast/strided_slice:output:0* 
_output_shapes
::
Erandom_contrast/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
Arandom_contrast/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV27random_contrast/stateless_random_uniform/shape:output:0Krandom_contrast/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Orandom_contrast/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Nrandom_contrast/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: Â
,random_contrast/stateless_random_uniform/subSub5random_contrast/stateless_random_uniform/max:output:05random_contrast/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Ò
,random_contrast/stateless_random_uniform/mulMulJrandom_contrast/stateless_random_uniform/StatelessRandomUniformV2:output:00random_contrast/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: »
(random_contrast/stateless_random_uniformAddV20random_contrast/stateless_random_uniform/mul:z:05random_contrast/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Å
random_contrast/adjust_contrastAdjustContrastv2/resizing/resize/ResizeBilinear:resized_images:0,random_contrast/stateless_random_uniform:z:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
(random_contrast/adjust_contrast/IdentityIdentity(random_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààz
batch_normalization/ShapeShape1random_contrast/adjust_contrast/Identity:output:0*
T0*
_output_shapes
:q
'batch_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)batch_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)batch_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!batch_normalization/strided_sliceStridedSlice"batch_normalization/Shape:output:00batch_normalization/strided_slice/stack:output:02batch_normalization/strided_slice/stack_1:output:02batch_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
batch_normalization/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 
batch_normalization/GreaterGreater*batch_normalization/strided_slice:output:0&batch_normalization/Greater/y:output:0*
T0*
_output_shapes
: å
batch_normalization/condIfbatch_normalization/Greater:z:0 batch_normalization_cond_input_0 batch_normalization_cond_input_1 batch_normalization_cond_input_2 batch_normalization_cond_input_31random_contrast/adjust_contrast/Identity:output:0*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*=
_output_shapes+
):ÿÿÿÿÿÿÿÿÿàà::*&
_read_only_resource_inputs
*9
else_branch*R(
&batch_normalization_cond_false_1442805*<
output_shapes+
):ÿÿÿÿÿÿÿÿÿàà::*8
then_branch)R'
%batch_normalization_cond_true_1442804
!batch_normalization/cond/IdentityIdentity!batch_normalization/cond:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààw
#batch_normalization/cond/Identity_1Identity!batch_normalization/cond:output:1*
T0*
_output_shapes
:w
#batch_normalization/cond/Identity_2Identity!batch_normalization/cond:output:2*
T0*
_output_shapes
:Ä
"batch_normalization/AssignNewValueAssignVariableOp batch_normalization_cond_input_2,batch_normalization/cond/Identity_1:output:0^batch_normalization/cond*
_output_shapes
 *
dtype0Æ
$batch_normalization/AssignNewValue_1AssignVariableOp batch_normalization_cond_input_3,batch_normalization/cond/Identity_2:output:0^batch_normalization/cond*
_output_shapes
 *
dtype0
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ì
conv2d/Conv2DConv2D*batch_normalization/cond/Identity:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnnj
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0Ã
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0Å
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0®
max_pooling2d/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
ksize
*
paddingVALID*
strides
i
batch_normalization_1/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:s
)batch_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+batch_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+batch_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#batch_normalization_1/strided_sliceStridedSlice$batch_normalization_1/Shape:output:02batch_normalization_1/strided_slice/stack:output:04batch_normalization_1/strided_slice/stack_1:output:04batch_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
batch_normalization_1/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ¡
batch_normalization_1/GreaterGreater,batch_normalization_1/strided_slice:output:0(batch_normalization_1/Greater/y:output:0*
T0*
_output_shapes
: Þ
batch_normalization_1/condIf!batch_normalization_1/Greater:z:0"batch_normalization_1_cond_input_0"batch_normalization_1_cond_input_1"batch_normalization_1_cond_input_2"batch_normalization_1_cond_input_3max_pooling2d/MaxPool:output:0*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*;
_output_shapes)
':ÿÿÿÿÿÿÿÿÿ0:0:0*&
_read_only_resource_inputs
*;
else_branch,R*
(batch_normalization_1_cond_false_1442876*:
output_shapes)
':ÿÿÿÿÿÿÿÿÿ0:0:0*:
then_branch+R)
'batch_normalization_1_cond_true_1442875
#batch_normalization_1/cond/IdentityIdentity#batch_normalization_1/cond:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0{
%batch_normalization_1/cond/Identity_1Identity#batch_normalization_1/cond:output:1*
T0*
_output_shapes
:0{
%batch_normalization_1/cond/Identity_2Identity#batch_normalization_1/cond:output:2*
T0*
_output_shapes
:0Ì
$batch_normalization_1/AssignNewValueAssignVariableOp"batch_normalization_1_cond_input_2.batch_normalization_1/cond/Identity_1:output:0^batch_normalization_1/cond*
_output_shapes
 *
dtype0Î
&batch_normalization_1/AssignNewValue_1AssignVariableOp"batch_normalization_1_cond_input_3.batch_normalization_1/cond/Identity_2:output:0^batch_normalization_1/cond*
_output_shapes
 *
dtype0
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0Ò
conv2d_3/Conv2DConv2D,batch_normalization_1/cond/Identity:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Å
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
max_pooling2d_1/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
activation_5/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?
dropout/dropout/MulMulactivation_5/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dropout/dropout/ShapeShapeactivation_5/Relu:activations:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >¿
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
activation_6/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_2/MatMulMatMulactivation_6/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1^batch_normalization/cond%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1^batch_normalization_1/cond^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp9^random_contrast/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_124
batch_normalization/condbatch_normalization/cond2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_128
batch_normalization_1/condbatch_normalization_1/cond2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2t
8random_contrast/stateful_uniform_full_int/RngReadAndSkip8random_contrast/stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs
ú	
c
D__inference_dropout_layer_call_and_return_conditional_losses_1443703

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ

*__inference_conv2d_1_layer_call_fn_1443255

inputs!
unknown:$
	unknown_0:$
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1441531w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿnn: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
 
_user_specified_nameinputs
ë
c
G__inference_activation_layer_call_and_return_conditional_losses_1443241

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnnb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿnn:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
 
_user_specified_nameinputs
°
J
.__inference_activation_5_layer_call_fn_1443671

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1441653a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_random_contrast_layer_call_and_return_conditional_losses_1443045

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿàà:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
Ù
É
cond_true_1443145*
cond_readvariableop_resource:,
cond_readvariableop_1_resource:;
-cond_fusedbatchnormv3_readvariableop_resource:=
/cond_fusedbatchnormv3_readvariableop_1_resource: 
cond_fusedbatchnormv3_inputs
cond_identity
cond_identity_1
cond_identity_2¢$cond/FusedBatchNormV3/ReadVariableOp¢&cond/FusedBatchNormV3/ReadVariableOp_1¢cond/ReadVariableOp¢cond/ReadVariableOp_1l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:*
dtype0
$cond/FusedBatchNormV3/ReadVariableOpReadVariableOp-cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
&cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
cond/FusedBatchNormV3FusedBatchNormV3cond_fusedbatchnormv3_inputscond/ReadVariableOp:value:0cond/ReadVariableOp_1:value:0,cond/FusedBatchNormV3/ReadVariableOp:value:0.cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<
cond/IdentityIdentitycond/FusedBatchNormV3:y:0
^cond/NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿp
cond/Identity_1Identity"cond/FusedBatchNormV3:batch_mean:0
^cond/NoOp*
T0*
_output_shapes
:t
cond/Identity_2Identity&cond/FusedBatchNormV3:batch_variance:0
^cond/NoOp*
T0*
_output_shapes
:É
	cond/NoOpNoOp%^cond/FusedBatchNormV3/ReadVariableOp'^cond/FusedBatchNormV3/ReadVariableOp_1^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2L
$cond/FusedBatchNormV3/ReadVariableOp$cond/FusedBatchNormV3/ReadVariableOp2P
&cond/FusedBatchNormV3/ReadVariableOp_1&cond/FusedBatchNormV3/ReadVariableOp_12*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
­
ï
%batch_normalization_cond_true_1442804>
0batch_normalization_cond_readvariableop_resource:@
2batch_normalization_cond_readvariableop_1_resource:O
Abatch_normalization_cond_fusedbatchnormv3_readvariableop_resource:Q
Cbatch_normalization_cond_fusedbatchnormv3_readvariableop_1_resource:V
Rbatch_normalization_cond_fusedbatchnormv3_random_contrast_adjust_contrast_identity%
!batch_normalization_cond_identity'
#batch_normalization_cond_identity_1'
#batch_normalization_cond_identity_2¢8batch_normalization/cond/FusedBatchNormV3/ReadVariableOp¢:batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1¢'batch_normalization/cond/ReadVariableOp¢)batch_normalization/cond/ReadVariableOp_1
'batch_normalization/cond/ReadVariableOpReadVariableOp0batch_normalization_cond_readvariableop_resource*
_output_shapes
:*
dtype0
)batch_normalization/cond/ReadVariableOp_1ReadVariableOp2batch_normalization_cond_readvariableop_1_resource*
_output_shapes
:*
dtype0¶
8batch_normalization/cond/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0º
:batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0
)batch_normalization/cond/FusedBatchNormV3FusedBatchNormV3Rbatch_normalization_cond_fusedbatchnormv3_random_contrast_adjust_contrast_identity/batch_normalization/cond/ReadVariableOp:value:01batch_normalization/cond/ReadVariableOp_1:value:0@batch_normalization/cond/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿàà:::::*
epsilon%o:*
exponential_avg_factor%
×#<¸
!batch_normalization/cond/IdentityIdentity-batch_normalization/cond/FusedBatchNormV3:y:0^batch_normalization/cond/NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà¬
#batch_normalization/cond/Identity_1Identity6batch_normalization/cond/FusedBatchNormV3:batch_mean:0^batch_normalization/cond/NoOp*
T0*
_output_shapes
:°
#batch_normalization/cond/Identity_2Identity:batch_normalization/cond/FusedBatchNormV3:batch_variance:0^batch_normalization/cond/NoOp*
T0*
_output_shapes
:­
batch_normalization/cond/NoOpNoOp9^batch_normalization/cond/FusedBatchNormV3/ReadVariableOp;^batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization/cond/ReadVariableOp*^batch_normalization/cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "O
!batch_normalization_cond_identity*batch_normalization/cond/Identity:output:0"S
#batch_normalization_cond_identity_1,batch_normalization/cond/Identity_1:output:0"S
#batch_normalization_cond_identity_2,batch_normalization/cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿàà2t
8batch_normalization/cond/FusedBatchNormV3/ReadVariableOp8batch_normalization/cond/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1:batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization/cond/ReadVariableOp'batch_normalization/cond/ReadVariableOp2V
)batch_normalization/cond/ReadVariableOp_1)batch_normalization/cond/ReadVariableOp_1:73
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà

þ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1443570

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç

)__inference_dense_2_layer_call_fn_1443741

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1441695o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç	
õ
D__inference_dense_2_layer_call_and_return_conditional_losses_1443751

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ù
É
cond_true_1441402*
cond_readvariableop_resource:0,
cond_readvariableop_1_resource:0;
-cond_fusedbatchnormv3_readvariableop_resource:0=
/cond_fusedbatchnormv3_readvariableop_1_resource:0 
cond_fusedbatchnormv3_inputs
cond_identity
cond_identity_1
cond_identity_2¢$cond/FusedBatchNormV3/ReadVariableOp¢&cond/FusedBatchNormV3/ReadVariableOp_1¢cond/ReadVariableOp¢cond/ReadVariableOp_1l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:0*
dtype0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:0*
dtype0
$cond/FusedBatchNormV3/ReadVariableOpReadVariableOp-cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
&cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
cond/FusedBatchNormV3FusedBatchNormV3cond_fusedbatchnormv3_inputscond/ReadVariableOp:value:0cond/ReadVariableOp_1:value:0,cond/FusedBatchNormV3/ReadVariableOp:value:0.cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<
cond/IdentityIdentitycond/FusedBatchNormV3:y:0
^cond/NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0p
cond/Identity_1Identity"cond/FusedBatchNormV3:batch_mean:0
^cond/NoOp*
T0*
_output_shapes
:0t
cond/Identity_2Identity&cond/FusedBatchNormV3:batch_variance:0
^cond/NoOp*
T0*
_output_shapes
:0É
	cond/NoOpNoOp%^cond/FusedBatchNormV3/ReadVariableOp'^cond/FusedBatchNormV3/ReadVariableOp_1^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02L
$cond/FusedBatchNormV3/ReadVariableOp$cond/FusedBatchNormV3/ReadVariableOp2P
&cond/FusedBatchNormV3/ReadVariableOp_1&cond/FusedBatchNormV3/ReadVariableOp_12*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
Ë	
ö
D__inference_dense_1_layer_call_and_return_conditional_losses_1441672

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

'__inference_model_layer_call_fn_1442596

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:$
	unknown_6:$#
	unknown_7:$0
	unknown_8:0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0$

unknown_13:0@

unknown_14:@$

unknown_15:@@

unknown_16:@

unknown_17:


unknown_18:	

unknown_19:	@

unknown_20:@

unknown_21:@

unknown_22:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1441702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs
¯
a
E__inference_resizing_layer_call_and_return_conditional_losses_1443029

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀð:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs
¤/
×
L__inference_random_contrast_layer_call_and_return_conditional_losses_1443086

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Ú
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:w
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0y
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ï
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	T

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R x
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_maska
stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL?a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Ì
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: 
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ¢
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: |
adjust_contrastAdjustContrastv2inputsstateless_random_uniform:z:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààz
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààz
IdentityIdentity!adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿààq
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿàà: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1443502

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
	
Ð
5__inference_batch_normalization_layer_call_fn_1443099

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1441263
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

J
.__inference_activation_1_layer_call_fn_1443294

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1442075z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
ÿ

R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1443421

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¯
a
E__inference_resizing_layer_call_and_return_conditional_losses_1441481

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀð:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs
ë
c
G__inference_activation_layer_call_and_return_conditional_losses_1441519

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnnb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿnn:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
 
_user_specified_nameinputs
Ù
É
cond_true_1443431*
cond_readvariableop_resource:0,
cond_readvariableop_1_resource:0;
-cond_fusedbatchnormv3_readvariableop_resource:0=
/cond_fusedbatchnormv3_readvariableop_1_resource:0 
cond_fusedbatchnormv3_inputs
cond_identity
cond_identity_1
cond_identity_2¢$cond/FusedBatchNormV3/ReadVariableOp¢&cond/FusedBatchNormV3/ReadVariableOp_1¢cond/ReadVariableOp¢cond/ReadVariableOp_1l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:0*
dtype0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:0*
dtype0
$cond/FusedBatchNormV3/ReadVariableOpReadVariableOp-cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0
&cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
cond/FusedBatchNormV3FusedBatchNormV3cond_fusedbatchnormv3_inputscond/ReadVariableOp:value:0cond/ReadVariableOp_1:value:0,cond/FusedBatchNormV3/ReadVariableOp:value:0.cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<
cond/IdentityIdentitycond/FusedBatchNormV3:y:0
^cond/NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0p
cond/Identity_1Identity"cond/FusedBatchNormV3:batch_mean:0
^cond/NoOp*
T0*
_output_shapes
:0t
cond/Identity_2Identity&cond/FusedBatchNormV3:batch_variance:0
^cond/NoOp*
T0*
_output_shapes
:0É
	cond/NoOpNoOp%^cond/FusedBatchNormV3/ReadVariableOp'^cond/FusedBatchNormV3/ReadVariableOp_1^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02L
$cond/FusedBatchNormV3/ReadVariableOp$cond/FusedBatchNormV3/ReadVariableOp2P
&cond/FusedBatchNormV3/ReadVariableOp_1&cond/FusedBatchNormV3/ReadVariableOp_12*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
í
ö
'batch_normalization_1_cond_true_1442875@
2batch_normalization_1_cond_readvariableop_resource:0B
4batch_normalization_1_cond_readvariableop_1_resource:0Q
Cbatch_normalization_1_cond_fusedbatchnormv3_readvariableop_resource:0S
Ebatch_normalization_1_cond_fusedbatchnormv3_readvariableop_1_resource:0E
Abatch_normalization_1_cond_fusedbatchnormv3_max_pooling2d_maxpool'
#batch_normalization_1_cond_identity)
%batch_normalization_1_cond_identity_1)
%batch_normalization_1_cond_identity_2¢:batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp¢<batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1¢)batch_normalization_1/cond/ReadVariableOp¢+batch_normalization_1/cond/ReadVariableOp_1
)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1_cond_readvariableop_resource*
_output_shapes
:0*
dtype0
+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1_cond_readvariableop_1_resource*
_output_shapes
:0*
dtype0º
:batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOpReadVariableOpCbatch_normalization_1_cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0¾
<batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEbatch_normalization_1_cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0
+batch_normalization_1/cond/FusedBatchNormV3FusedBatchNormV3Abatch_normalization_1_cond_fusedbatchnormv3_max_pooling2d_maxpool1batch_normalization_1/cond/ReadVariableOp:value:03batch_normalization_1/cond/ReadVariableOp_1:value:0Bbatch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp:value:0Dbatch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
exponential_avg_factor%
×#<¼
#batch_normalization_1/cond/IdentityIdentity/batch_normalization_1/cond/FusedBatchNormV3:y:0 ^batch_normalization_1/cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0²
%batch_normalization_1/cond/Identity_1Identity8batch_normalization_1/cond/FusedBatchNormV3:batch_mean:0 ^batch_normalization_1/cond/NoOp*
T0*
_output_shapes
:0¶
%batch_normalization_1/cond/Identity_2Identity<batch_normalization_1/cond/FusedBatchNormV3:batch_variance:0 ^batch_normalization_1/cond/NoOp*
T0*
_output_shapes
:0·
batch_normalization_1/cond/NoOpNoOp;^batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp=^batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1*^batch_normalization_1/cond/ReadVariableOp,^batch_normalization_1/cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "S
#batch_normalization_1_cond_identity,batch_normalization_1/cond/Identity:output:0"W
%batch_normalization_1_cond_identity_1.batch_normalization_1/cond/Identity_1:output:0"W
%batch_normalization_1_cond_identity_2.batch_normalization_1/cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :ÿÿÿÿÿÿÿÿÿ02x
:batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp:batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp2|
<batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1<batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_12V
)batch_normalization_1/cond/ReadVariableOp)batch_normalization_1/cond/ReadVariableOp2Z
+batch_normalization_1/cond/ReadVariableOp_1+batch_normalization_1/cond/ReadVariableOp_1:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
³
c
G__inference_activation_layer_call_and_return_conditional_losses_1443246

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
e
I__inference_activation_4_layer_call_and_return_conditional_losses_1443590

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1441587

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1441531

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿnn: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
 
_user_specified_nameinputs
Ê

'__inference_dense_layer_call_fn_1443637

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1441642p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

J
.__inference_activation_3_layer_call_fn_1443522

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1442127z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1443274

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿnn: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
 
_user_specified_nameinputs
µ
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1443304

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
í
e
I__inference_activation_4_layer_call_and_return_conditional_losses_1441621

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ñ

*__inference_conv2d_3_layer_call_fn_1443483

inputs!
unknown:0@
	unknown_0:@
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1441587w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
Í
e
I__inference_activation_6_layer_call_and_return_conditional_losses_1443732

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
µ
e
I__inference_activation_1_layer_call_and_return_conditional_losses_1442075

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
ÿ|
É
B__inference_model_layer_call_and_return_conditional_losses_1442755

inputs9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:$6
(conv2d_1_biasadd_readvariableop_resource:$A
'conv2d_2_conv2d_readvariableop_resource:$06
(conv2d_2_biasadd_readvariableop_resource:0;
-batch_normalization_1_readvariableop_resource:0=
/batch_normalization_1_readvariableop_1_resource:0L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:0A
'conv2d_3_conv2d_readvariableop_resource:0@6
(conv2d_3_biasadd_readvariableop_resource:@A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@8
$dense_matmul_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity¢3batch_normalization/FusedBatchNormV3/ReadVariableOp¢5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¢"batch_normalization/ReadVariableOp¢$batch_normalization/ReadVariableOp_1¢5batch_normalization_1/FusedBatchNormV3/ReadVariableOp¢7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1¢$batch_normalization_1/ReadVariableOp¢&batch_normalization_1/ReadVariableOp_1¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢dense_2/BiasAdd/ReadVariableOp¢dense_2/MatMul/ReadVariableOpe
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"à   à   ­
resizing/resize/ResizeBilinearResizeBilinearinputsresizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*
half_pixel_centers(x
batch_normalization/ShapeShape/resizing/resize/ResizeBilinear:resized_images:0*
T0*
_output_shapes
:q
'batch_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)batch_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)batch_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!batch_normalization/strided_sliceStridedSlice"batch_normalization/Shape:output:00batch_normalization/strided_slice/stack:output:02batch_normalization/strided_slice/stack_1:output:02batch_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0¬
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0°
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Å
$batch_normalization/FusedBatchNormV3FusedBatchNormV3/resizing/resize/ResizeBilinear:resized_images:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿàà:::::*
epsilon%o:*
is_training( 
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ê
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnnj
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0Ã
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0Å
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0®
max_pooling2d/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
ksize
*
paddingVALID*
strides
i
batch_normalization_1/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:s
)batch_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+batch_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+batch_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#batch_normalization_1/strided_sliceStridedSlice$batch_normalization_1/Shape:output:02batch_normalization_1/strided_slice/stack:output:04batch_normalization_1/strided_slice/stack_1:output:04batch_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype0
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype0°
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0´
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0¼
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ0:0:0:0:0:*
epsilon%o:*
is_training( 
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0Ð
conv2d_3/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*
paddingVALID*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Å
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@n
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@°
max_pooling2d_1/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
activation_5/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/IdentityIdentityactivation_5/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
activation_6/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
dense_2/MatMulMatMulactivation_6/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs
³
c
G__inference_activation_layer_call_and_return_conditional_losses_1442054

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿt
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«

ü
C__inference_conv2d_layer_call_and_return_conditional_losses_1441508

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnng
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnnw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs
º

*__inference_conv2d_2_layer_call_fn_1443322

inputs!
unknown:$0
	unknown_0:0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1442086
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
 
_user_specified_nameinputs
Ð

1__inference_random_contrast_layer_call_fn_1443041

inputs
unknown:	
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_1441953y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿàà: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

¤
cond_false_1443146
cond_placeholder
cond_placeholder_1*
cond_readvariableop_resource:,
cond_readvariableop_1_resource:
cond_identity_inputs
cond_identity
cond_identity_1
cond_identity_2¢cond/ReadVariableOp¢cond/ReadVariableOp_1
cond/IdentityIdentitycond_identity_inputs
^cond/NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿl
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype0i
cond/Identity_1Identitycond/ReadVariableOp:value:0
^cond/NoOp*
T0*
_output_shapes
:p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:*
dtype0k
cond/Identity_2Identitycond/ReadVariableOp_1:value:0
^cond/NoOp*
T0*
_output_shapes
:y
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
©

þ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1441554

inputs8
conv2d_readvariableop_resource:$0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ55$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$
 
_user_specified_nameinputs
Ç	
õ
D__inference_dense_2_layer_call_and_return_conditional_losses_1441695

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
à	
ö
B__inference_dense_layer_call_and_return_conditional_losses_1442173

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«

ü
C__inference_conv2d_layer_call_and_return_conditional_losses_1443216

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnng
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿnnw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿàà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1443372

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

h
L__inference_random_contrast_layer_call_and_return_conditional_losses_1441487

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿàà:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà
 
_user_specified_nameinputs

H
,__inference_activation_layer_call_fn_1443236

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1442054z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
ª
'__inference_model_layer_call_fn_1442385
input_1
unknown:	
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:#
	unknown_6:$
	unknown_7:$#
	unknown_8:$0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0@

unknown_15:@$

unknown_16:@@

unknown_17:@

unknown_18:


unknown_19:	

unknown_20:	@

unknown_21:@

unknown_22:@

unknown_23:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1442193o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
!
_user_specified_name	input_1

¤
cond_false_1441403
cond_placeholder
cond_placeholder_1*
cond_readvariableop_resource:0,
cond_readvariableop_1_resource:0
cond_identity_inputs
cond_identity
cond_identity_1
cond_identity_2¢cond/ReadVariableOp¢cond/ReadVariableOp_1
cond/IdentityIdentitycond_identity_inputs
^cond/NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:0*
dtype0i
cond/Identity_1Identitycond/ReadVariableOp:value:0
^cond/NoOp*
T0*
_output_shapes
:0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:0*
dtype0k
cond/Identity_2Identitycond/ReadVariableOp_1:value:0
^cond/NoOp*
T0*
_output_shapes
:0y
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ02*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
Î
©
'__inference_model_layer_call_fn_1442651

inputs
unknown:	
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:#
	unknown_6:$
	unknown_7:$#
	unknown_8:$0
	unknown_9:0

unknown_10:0

unknown_11:0

unknown_12:0

unknown_13:0$

unknown_14:0@

unknown_15:@$

unknown_16:@@

unknown_17:@

unknown_18:


unknown_19:	

unknown_20:	@

unknown_21:@

unknown_22:@

unknown_23:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_23*%
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1442193o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
 
_user_specified_nameinputs

¤
cond_false_1441287
cond_placeholder
cond_placeholder_1*
cond_readvariableop_resource:,
cond_readvariableop_1_resource:
cond_identity_inputs
cond_identity
cond_identity_1
cond_identity_2¢cond/ReadVariableOp¢cond/ReadVariableOp_1
cond/IdentityIdentitycond_identity_inputs
^cond/NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿl
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype0i
cond/Identity_1Identitycond/ReadVariableOp:value:0
^cond/NoOp*
T0*
_output_shapes
:p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:*
dtype0k
cond/Identity_2Identitycond/ReadVariableOp_1:value:0
^cond/NoOp*
T0*
_output_shapes
:y
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¯

`
D__inference_flatten_layer_call_and_return_conditional_losses_1443628

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿu
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿa
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ì
J
.__inference_activation_2_layer_call_fn_1443347

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1441565h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ0:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
¦
E
)__inference_dropout_layer_call_fn_1443681

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1441660a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
b
)__inference_dropout_layer_call_fn_1443686

inputs
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1441799p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

)__inference_dense_1_layer_call_fn_1443712

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1441672o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
e
I__inference_activation_3_layer_call_and_return_conditional_losses_1441598

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ

@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
 
_user_specified_nameinputs
¹
¸
 __inference__traced_save_1444001
file_prefix$
 savev2_gamma_read_readvariableop#
savev2_beta_read_readvariableop*
&savev2_moving_mean_read_readvariableop.
*savev2_moving_variance_read_readvariableop%
!savev2_kernel_read_readvariableop#
savev2_bias_read_readvariableop'
#savev2_kernel_1_read_readvariableop%
!savev2_bias_1_read_readvariableop'
#savev2_kernel_2_read_readvariableop%
!savev2_bias_2_read_readvariableop&
"savev2_gamma_1_read_readvariableop%
!savev2_beta_1_read_readvariableop,
(savev2_moving_mean_1_read_readvariableop0
,savev2_moving_variance_1_read_readvariableop'
#savev2_kernel_3_read_readvariableop%
!savev2_bias_3_read_readvariableop'
#savev2_kernel_4_read_readvariableop%
!savev2_bias_4_read_readvariableop'
#savev2_kernel_5_read_readvariableop%
!savev2_bias_5_read_readvariableop'
#savev2_kernel_6_read_readvariableop%
!savev2_bias_6_read_readvariableop'
#savev2_kernel_7_read_readvariableop%
!savev2_bias_7_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop7
3savev2_random_contrast_statevar_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop+
'savev2_adam_gamma_m_read_readvariableop*
&savev2_adam_beta_m_read_readvariableop,
(savev2_adam_kernel_m_read_readvariableop*
&savev2_adam_bias_m_read_readvariableop.
*savev2_adam_kernel_m_1_read_readvariableop,
(savev2_adam_bias_m_1_read_readvariableop.
*savev2_adam_kernel_m_2_read_readvariableop,
(savev2_adam_bias_m_2_read_readvariableop-
)savev2_adam_gamma_m_1_read_readvariableop,
(savev2_adam_beta_m_1_read_readvariableop.
*savev2_adam_kernel_m_3_read_readvariableop,
(savev2_adam_bias_m_3_read_readvariableop.
*savev2_adam_kernel_m_4_read_readvariableop,
(savev2_adam_bias_m_4_read_readvariableop.
*savev2_adam_kernel_m_5_read_readvariableop,
(savev2_adam_bias_m_5_read_readvariableop.
*savev2_adam_kernel_m_6_read_readvariableop,
(savev2_adam_bias_m_6_read_readvariableop.
*savev2_adam_kernel_m_7_read_readvariableop,
(savev2_adam_bias_m_7_read_readvariableop+
'savev2_adam_gamma_v_read_readvariableop*
&savev2_adam_beta_v_read_readvariableop,
(savev2_adam_kernel_v_read_readvariableop*
&savev2_adam_bias_v_read_readvariableop.
*savev2_adam_kernel_v_1_read_readvariableop,
(savev2_adam_bias_v_1_read_readvariableop.
*savev2_adam_kernel_v_2_read_readvariableop,
(savev2_adam_bias_v_2_read_readvariableop-
)savev2_adam_gamma_v_1_read_readvariableop,
(savev2_adam_beta_v_1_read_readvariableop.
*savev2_adam_kernel_v_3_read_readvariableop,
(savev2_adam_bias_v_3_read_readvariableop.
*savev2_adam_kernel_v_4_read_readvariableop,
(savev2_adam_bias_v_4_read_readvariableop.
*savev2_adam_kernel_v_5_read_readvariableop,
(savev2_adam_bias_v_5_read_readvariableop.
*savev2_adam_kernel_v_6_read_readvariableop,
(savev2_adam_bias_v_6_read_readvariableop.
*savev2_adam_kernel_v_7_read_readvariableop,
(savev2_adam_bias_v_7_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Í)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*ö(
valueì(Bé(KB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*«
value¡BKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_gamma_read_readvariableopsavev2_beta_read_readvariableop&savev2_moving_mean_read_readvariableop*savev2_moving_variance_read_readvariableop!savev2_kernel_read_readvariableopsavev2_bias_read_readvariableop#savev2_kernel_1_read_readvariableop!savev2_bias_1_read_readvariableop#savev2_kernel_2_read_readvariableop!savev2_bias_2_read_readvariableop"savev2_gamma_1_read_readvariableop!savev2_beta_1_read_readvariableop(savev2_moving_mean_1_read_readvariableop,savev2_moving_variance_1_read_readvariableop#savev2_kernel_3_read_readvariableop!savev2_bias_3_read_readvariableop#savev2_kernel_4_read_readvariableop!savev2_bias_4_read_readvariableop#savev2_kernel_5_read_readvariableop!savev2_bias_5_read_readvariableop#savev2_kernel_6_read_readvariableop!savev2_bias_6_read_readvariableop#savev2_kernel_7_read_readvariableop!savev2_bias_7_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop3savev2_random_contrast_statevar_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop'savev2_adam_gamma_m_read_readvariableop&savev2_adam_beta_m_read_readvariableop(savev2_adam_kernel_m_read_readvariableop&savev2_adam_bias_m_read_readvariableop*savev2_adam_kernel_m_1_read_readvariableop(savev2_adam_bias_m_1_read_readvariableop*savev2_adam_kernel_m_2_read_readvariableop(savev2_adam_bias_m_2_read_readvariableop)savev2_adam_gamma_m_1_read_readvariableop(savev2_adam_beta_m_1_read_readvariableop*savev2_adam_kernel_m_3_read_readvariableop(savev2_adam_bias_m_3_read_readvariableop*savev2_adam_kernel_m_4_read_readvariableop(savev2_adam_bias_m_4_read_readvariableop*savev2_adam_kernel_m_5_read_readvariableop(savev2_adam_bias_m_5_read_readvariableop*savev2_adam_kernel_m_6_read_readvariableop(savev2_adam_bias_m_6_read_readvariableop*savev2_adam_kernel_m_7_read_readvariableop(savev2_adam_bias_m_7_read_readvariableop'savev2_adam_gamma_v_read_readvariableop&savev2_adam_beta_v_read_readvariableop(savev2_adam_kernel_v_read_readvariableop&savev2_adam_bias_v_read_readvariableop*savev2_adam_kernel_v_1_read_readvariableop(savev2_adam_bias_v_1_read_readvariableop*savev2_adam_kernel_v_2_read_readvariableop(savev2_adam_bias_v_2_read_readvariableop)savev2_adam_gamma_v_1_read_readvariableop(savev2_adam_beta_v_1_read_readvariableop*savev2_adam_kernel_v_3_read_readvariableop(savev2_adam_bias_v_3_read_readvariableop*savev2_adam_kernel_v_4_read_readvariableop(savev2_adam_bias_v_4_read_readvariableop*savev2_adam_kernel_v_5_read_readvariableop(savev2_adam_bias_v_5_read_readvariableop*savev2_adam_kernel_v_6_read_readvariableop(savev2_adam_bias_v_6_read_readvariableop*savev2_adam_kernel_v_7_read_readvariableop(savev2_adam_bias_v_7_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: :::::::$:$:$0:0:0:0:0:0:0@:@:@@:@:
::	@:@:@:: : : : : :: : : : :::::$:$:$0:0:0:0:0@:@:@@:@:
::	@:@:@::::::$:$:$0:0:0:0:0@:@:@@:@:
::	@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:$: 

_output_shapes
:$:,	(
&
_output_shapes
:$0: 


_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
::

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: : #

_output_shapes
:: $

_output_shapes
::,%(
&
_output_shapes
:: &

_output_shapes
::,'(
&
_output_shapes
:$: (

_output_shapes
:$:,)(
&
_output_shapes
:$0: *

_output_shapes
:0: +

_output_shapes
:0: ,

_output_shapes
:0:,-(
&
_output_shapes
:0@: .

_output_shapes
:@:,/(
&
_output_shapes
:@@: 0

_output_shapes
:@:&1"
 
_output_shapes
:
:!2

_output_shapes	
::%3!

_output_shapes
:	@: 4

_output_shapes
:@:$5 

_output_shapes

:@: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
::,9(
&
_output_shapes
:: :

_output_shapes
::,;(
&
_output_shapes
:$: <

_output_shapes
:$:,=(
&
_output_shapes
:$0: >

_output_shapes
:0: ?

_output_shapes
:0: @

_output_shapes
:0:,A(
&
_output_shapes
:0@: B

_output_shapes
:@:,C(
&
_output_shapes
:@@: D

_output_shapes
:@:&E"
 
_output_shapes
:
:!F

_output_shapes	
::%G!

_output_shapes
:	@: H

_output_shapes
:@:$I 

_output_shapes

:@: J

_output_shapes
::K

_output_shapes
: 
Ð	
ö
B__inference_dense_layer_call_and_return_conditional_losses_1441642

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ü
C__inference_conv2d_layer_call_and_return_conditional_losses_1443226

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿy
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý

P__inference_batch_normalization_layer_call_and_return_conditional_losses_1443135

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
J
.__inference_activation_6_layer_call_fn_1443727

inputs
identity¹
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1441683`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Ò
7__inference_batch_normalization_1_layer_call_fn_1443385

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1441379
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
_user_specified_nameinputs
©

þ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1441610

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ

@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@
 
_user_specified_nameinputs
í
e
I__inference_activation_4_layer_call_and_return_conditional_losses_1443585

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
^
ï

B__inference_model_layer_call_and_return_conditional_losses_1442460
input_1)
batch_normalization_1442390:)
batch_normalization_1442392:)
batch_normalization_1442394:)
batch_normalization_1442396:(
conv2d_1442399:
conv2d_1442401:*
conv2d_1_1442405:$
conv2d_1_1442407:$*
conv2d_2_1442411:$0
conv2d_2_1442413:0+
batch_normalization_1_1442418:0+
batch_normalization_1_1442420:0+
batch_normalization_1_1442422:0+
batch_normalization_1_1442424:0*
conv2d_3_1442427:0@
conv2d_3_1442429:@*
conv2d_4_1442433:@@
conv2d_4_1442435:@!
dense_1442441:

dense_1442443:	"
dense_1_1442448:	@
dense_1_1442450:@!
dense_2_1442454:@
dense_2_1442456:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallÉ
resizing/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_resizing_layer_call_and_return_conditional_losses_1441481ñ
random_contrast/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_1441487
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(random_contrast/PartitionedCall:output:0batch_normalization_1442390batch_normalization_1442392batch_normalization_1442394batch_normalization_1442396*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿàà*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1441263¦
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1442399conv2d_1442401*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_1441508ë
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿnn* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_1441519
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_1442405conv2d_1_1442407*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1441531ñ
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ55$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_1441542
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_1442411conv2d_2_1442413*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1441554ñ
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_1441565ï
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1441349
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_1442418batch_normalization_1_1442420batch_normalization_1_1442422batch_normalization_1_1442424*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1441379°
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_3_1442427conv2d_3_1442429*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1441587ñ
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_1441598
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_1442433conv2d_4_1442435*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1441610ñ
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_1441621ó
max_pooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1441465ß
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_1441630
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1442441dense_1442443*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1441642ç
activation_5/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_1441653Ü
dropout/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_1441660
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_1442448dense_1_1442450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_1441672è
activation_6/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_1441683
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_2_1442454dense_2_1442456*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_1441695w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:ÿÿÿÿÿÿÿÿÿÀð: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀð
!
_user_specified_name	input_1

¥
(batch_normalization_1_cond_false_1442876*
&batch_normalization_1_cond_placeholder,
(batch_normalization_1_cond_placeholder_1@
2batch_normalization_1_cond_readvariableop_resource:0B
4batch_normalization_1_cond_readvariableop_1_resource:0=
9batch_normalization_1_cond_identity_max_pooling2d_maxpool'
#batch_normalization_1_cond_identity)
%batch_normalization_1_cond_identity_1)
%batch_normalization_1_cond_identity_2¢)batch_normalization_1/cond/ReadVariableOp¢+batch_normalization_1/cond/ReadVariableOp_1Æ
#batch_normalization_1/cond/IdentityIdentity9batch_normalization_1_cond_identity_max_pooling2d_maxpool ^batch_normalization_1/cond/NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0
)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1_cond_readvariableop_resource*
_output_shapes
:0*
dtype0«
%batch_normalization_1/cond/Identity_1Identity1batch_normalization_1/cond/ReadVariableOp:value:0 ^batch_normalization_1/cond/NoOp*
T0*
_output_shapes
:0
+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1_cond_readvariableop_1_resource*
_output_shapes
:0*
dtype0­
%batch_normalization_1/cond/Identity_2Identity3batch_normalization_1/cond/ReadVariableOp_1:value:0 ^batch_normalization_1/cond/NoOp*
T0*
_output_shapes
:0»
batch_normalization_1/cond/NoOpNoOp*^batch_normalization_1/cond/ReadVariableOp,^batch_normalization_1/cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "S
#batch_normalization_1_cond_identity,batch_normalization_1/cond/Identity:output:0"W
%batch_normalization_1_cond_identity_1.batch_normalization_1/cond/Identity_1:output:0"W
%batch_normalization_1_cond_identity_2.batch_normalization_1/cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :ÿÿÿÿÿÿÿÿÿ02V
)batch_normalization_1/cond/ReadVariableOp)batch_normalization_1/cond/ReadVariableOp2Z
+batch_normalization_1/cond/ReadVariableOp_1+batch_normalization_1/cond/ReadVariableOp_1:51
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ0

þ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1443284

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0¬
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*´
serving_default 
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿÀð;
dense_20
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¯
Ò
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer-15
layer-16
layer-17
layer_with_weights-7
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer-22
layer_with_weights-9
layer-23
	optimizer

signatures
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature"
_tf_keras_network
D
##_self_saveable_object_factories"
_tf_keras_input_layer
Ê
#$_self_saveable_object_factories
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
á
#+_self_saveable_object_factories
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer

3axis
	4gamma
5beta
6moving_mean
7moving_variance
#8_self_saveable_object_factories
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
à

?kernel
@bias
#A_self_saveable_object_factories
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#H_self_saveable_object_factories
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
à

Okernel
Pbias
#Q_self_saveable_object_factories
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#X_self_saveable_object_factories
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
à

_kernel
`bias
#a_self_saveable_object_factories
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#h_self_saveable_object_factories
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
Ê
#o_self_saveable_object_factories
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer

vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
#{_self_saveable_object_factories
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
é
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
é
kernel
	bias
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
$_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
$¢_self_saveable_object_factories
£	variables
¤trainable_variables
¥regularization_losses
¦	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
$©_self_saveable_object_factories
ª	variables
«trainable_variables
¬regularization_losses
­	keras_api
®__call__
+¯&call_and_return_all_conditional_losses"
_tf_keras_layer
é
°kernel
	±bias
$²_self_saveable_object_factories
³	variables
´trainable_variables
µregularization_losses
¶	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
$¹_self_saveable_object_factories
º	variables
»trainable_variables
¼regularization_losses
½	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"
_tf_keras_layer
é
$À_self_saveable_object_factories
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å_random_generator
Æ__call__
+Ç&call_and_return_all_conditional_losses"
_tf_keras_layer
é
Èkernel
	Ébias
$Ê_self_saveable_object_factories
Ë	variables
Ìtrainable_variables
Íregularization_losses
Î	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_tf_keras_layer
Ñ
$Ñ_self_saveable_object_factories
Ò	variables
Ótrainable_variables
Ôregularization_losses
Õ	keras_api
Ö__call__
+×&call_and_return_all_conditional_losses"
_tf_keras_layer
é
Økernel
	Ùbias
$Ú_self_saveable_object_factories
Û	variables
Ütrainable_variables
Ýregularization_losses
Þ	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"
_tf_keras_layer
ü
	áiter
âbeta_1
ãbeta_2

ädecay
ålearning_rate4më5mì?mí@mîOmïPmð_mñ`mòwmóxmô	mõ	mö	m÷	mø	°mù	±mú	Èmû	Émü	Ømý	Ùmþ4vÿ5v?v@vOvPv_v`vwvxv	v	v	v	v	°v	±v	Èv	Év	Øv	Ùv"
	optimizer
-
æserving_default"
signature_map
 "
trackable_dict_wrapper
à
40
51
62
73
?4
@5
O6
P7
_8
`9
w10
x11
y12
z13
14
15
16
17
°18
±19
È20
É21
Ø22
Ù23"
trackable_list_wrapper
À
40
51
?2
@3
O4
P5
_6
`7
w8
x9
10
11
12
13
°14
±15
È16
É17
Ø18
Ù19"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
çnon_trainable_variables
èlayers
émetrics
 êlayer_regularization_losses
ëlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ê2ç
'__inference_model_layer_call_fn_1441753
'__inference_model_layer_call_fn_1442596
'__inference_model_layer_call_fn_1442651
'__inference_model_layer_call_fn_1442385À
·²³
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
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_model_layer_call_and_return_conditional_losses_1442755
B__inference_model_layer_call_and_return_conditional_losses_1442963
B__inference_model_layer_call_and_return_conditional_losses_1442460
B__inference_model_layer_call_and_return_conditional_losses_1442537À
·²³
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
kwonlydefaultsª 
annotationsª *
 
ÍBÊ
"__inference__wrapped_model_1441236input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
Ô2Ñ
*__inference_resizing_layer_call_fn_1443023¢
²
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
annotationsª *
 
ï2ì
E__inference_resizing_layer_call_and_return_conditional_losses_1443029¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ñnon_trainable_variables
òlayers
ómetrics
 ôlayer_regularization_losses
õlayer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
/
ö
_generator"
_generic_user_object
 2
1__inference_random_contrast_layer_call_fn_1443034
1__inference_random_contrast_layer_call_fn_1443041´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
L__inference_random_contrast_layer_call_and_return_conditional_losses_1443045
L__inference_random_contrast_layer_call_and_return_conditional_losses_1443086´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
: 2gamma
: 2beta
: (2moving_mean
: (2moving_variance
 "
trackable_dict_wrapper
<
40
51
62
73"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
²
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
¨2¥
5__inference_batch_normalization_layer_call_fn_1443099
5__inference_batch_normalization_layer_call_fn_1443112´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1443135
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1443188´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
":  2kernel
: 2bias
 "
trackable_dict_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ü2ù
(__inference_conv2d_layer_call_fn_1443197
(__inference_conv2d_layer_call_fn_1443206¢
²
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
annotationsª *
 
²2¯
C__inference_conv2d_layer_call_and_return_conditional_losses_1443216
C__inference_conv2d_layer_call_and_return_conditional_losses_1443226¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
2
,__inference_activation_layer_call_fn_1443231
,__inference_activation_layer_call_fn_1443236¢
²
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
annotationsª *
 
º2·
G__inference_activation_layer_call_and_return_conditional_losses_1443241
G__inference_activation_layer_call_and_return_conditional_losses_1443246¢
²
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
annotationsª *
 
": $ 2kernel
:$ 2bias
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
2ý
*__inference_conv2d_1_layer_call_fn_1443255
*__inference_conv2d_1_layer_call_fn_1443264¢
²
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
annotationsª *
 
¶2³
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1443274
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1443284¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_activation_1_layer_call_fn_1443289
.__inference_activation_1_layer_call_fn_1443294¢
²
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
annotationsª *
 
¾2»
I__inference_activation_1_layer_call_and_return_conditional_losses_1443299
I__inference_activation_1_layer_call_and_return_conditional_losses_1443304¢
²
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
annotationsª *
 
": $0 2kernel
:0 2bias
 "
trackable_dict_wrapper
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
2ý
*__inference_conv2d_2_layer_call_fn_1443313
*__inference_conv2d_2_layer_call_fn_1443322¢
²
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
annotationsª *
 
¶2³
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1443332
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1443342¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_activation_2_layer_call_fn_1443347
.__inference_activation_2_layer_call_fn_1443352¢
²
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
annotationsª *
 
¾2»
I__inference_activation_2_layer_call_and_return_conditional_losses_1443357
I__inference_activation_2_layer_call_and_return_conditional_losses_1443362¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_max_pooling2d_layer_call_fn_1443367¢
²
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
annotationsª *
 
ô2ñ
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1443372¢
²
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
annotationsª *
 
 "
trackable_list_wrapper
:0 2gamma
:0 2beta
:0 (2moving_mean
:0 (2moving_variance
 "
trackable_dict_wrapper
<
w0
x1
y2
z3"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¬2©
7__inference_batch_normalization_1_layer_call_fn_1443385
7__inference_batch_normalization_1_layer_call_fn_1443398´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1443421
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1443474´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
": 0@ 2kernel
:@ 2bias
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2ý
*__inference_conv2d_3_layer_call_fn_1443483
*__inference_conv2d_3_layer_call_fn_1443492¢
²
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
annotationsª *
 
¶2³
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1443502
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1443512¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_activation_3_layer_call_fn_1443517
.__inference_activation_3_layer_call_fn_1443522¢
²
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
annotationsª *
 
¾2»
I__inference_activation_3_layer_call_and_return_conditional_losses_1443527
I__inference_activation_3_layer_call_and_return_conditional_losses_1443532¢
²
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
annotationsª *
 
": @@ 2kernel
:@ 2bias
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2ý
*__inference_conv2d_4_layer_call_fn_1443541
*__inference_conv2d_4_layer_call_fn_1443550¢
²
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
annotationsª *
 
¶2³
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1443560
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1443570¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_activation_4_layer_call_fn_1443575
.__inference_activation_4_layer_call_fn_1443580¢
²
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
annotationsª *
 
¾2»
I__inference_activation_4_layer_call_and_return_conditional_losses_1443585
I__inference_activation_4_layer_call_and_return_conditional_losses_1443590¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
£	variables
¤trainable_variables
¥regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
Û2Ø
1__inference_max_pooling2d_1_layer_call_fn_1443595¢
²
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
annotationsª *
 
ö2ó
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1443600¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
ª	variables
«trainable_variables
¬regularization_losses
®__call__
+¯&call_and_return_all_conditional_losses
'¯"call_and_return_conditional_losses"
_generic_user_object
þ2û
)__inference_flatten_layer_call_fn_1443605
)__inference_flatten_layer_call_fn_1443610¢
²
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
annotationsª *
 
´2±
D__inference_flatten_layer_call_and_return_conditional_losses_1443616
D__inference_flatten_layer_call_and_return_conditional_losses_1443628¢
²
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
annotationsª *
 
:
 2kernel
: 2bias
 "
trackable_dict_wrapper
0
°0
±1"
trackable_list_wrapper
0
°0
±1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
³	variables
´trainable_variables
µregularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
ú2÷
'__inference_dense_layer_call_fn_1443637
'__inference_dense_layer_call_fn_1443646¢
²
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
annotationsª *
 
°2­
B__inference_dense_layer_call_and_return_conditional_losses_1443656
B__inference_dense_layer_call_and_return_conditional_losses_1443666¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
º	variables
»trainable_variables
¼regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_activation_5_layer_call_fn_1443671¢
²
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
annotationsª *
 
ó2ð
I__inference_activation_5_layer_call_and_return_conditional_losses_1443676¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
2
)__inference_dropout_layer_call_fn_1443681
)__inference_dropout_layer_call_fn_1443686´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_layer_call_and_return_conditional_losses_1443691
D__inference_dropout_layer_call_and_return_conditional_losses_1443703´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	@ 2kernel
:@ 2bias
 "
trackable_dict_wrapper
0
È0
É1"
trackable_list_wrapper
0
È0
É1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
Ë	variables
Ìtrainable_variables
Íregularization_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_1_layer_call_fn_1443712¢
²
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
annotationsª *
 
î2ë
D__inference_dense_1_layer_call_and_return_conditional_losses_1443722¢
²
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
annotationsª *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
Ò	variables
Ótrainable_variables
Ôregularization_losses
Ö__call__
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
Ø2Õ
.__inference_activation_6_layer_call_fn_1443727¢
²
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
annotationsª *
 
ó2ð
I__inference_activation_6_layer_call_and_return_conditional_losses_1443732¢
²
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
annotationsª *
 
:@ 2kernel
: 2bias
 "
trackable_dict_wrapper
0
Ø0
Ù1"
trackable_list_wrapper
0
Ø0
Ù1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
Û	variables
Ütrainable_variables
Ýregularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_dense_2_layer_call_fn_1443741¢
²
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
annotationsª *
 
î2ë
D__inference_dense_2_layer_call_and_return_conditional_losses_1443751¢
²
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
annotationsª *
 
:	  (2	Adam/iter
:  (2Adam/beta_1
:  (2Adam/beta_2
:  (2
Adam/decay
:  (2Adam/learning_rate
ÌBÉ
%__inference_signature_wrapper_1443018input_1"
²
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
annotationsª *
 
<
60
71
y2
z3"
trackable_list_wrapper
Ö
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
18
19
20
21
22
23"
trackable_list_wrapper
0
à0
á1"
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
/
â
_state_var"
_generic_user_object
.
60
71"
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
.
y0
z1"
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
R

ãtotal

äcount
å	variables
æ	keras_api"
_tf_keras_metric
R

çtotal

ècount
é	variables
ê	keras_api"
_tf_keras_metric
&:$	 2random_contrast/StateVar
:  (2total
:  (2count
0
ã0
ä1"
trackable_list_wrapper
.
å	variables"
_generic_user_object
:  (2total
:  (2count
0
ç0
è1"
trackable_list_wrapper
.
é	variables"
_generic_user_object
: 2Adam/gamma/m
: 2Adam/beta/m
':% 2Adam/kernel/m
: 2Adam/bias/m
':%$ 2Adam/kernel/m
:$ 2Adam/bias/m
':%$0 2Adam/kernel/m
:0 2Adam/bias/m
:0 2Adam/gamma/m
:0 2Adam/beta/m
':%0@ 2Adam/kernel/m
:@ 2Adam/bias/m
':%@@ 2Adam/kernel/m
:@ 2Adam/bias/m
!:
 2Adam/kernel/m
: 2Adam/bias/m
 :	@ 2Adam/kernel/m
:@ 2Adam/bias/m
:@ 2Adam/kernel/m
: 2Adam/bias/m
: 2Adam/gamma/v
: 2Adam/beta/v
':% 2Adam/kernel/v
: 2Adam/bias/v
':%$ 2Adam/kernel/v
:$ 2Adam/bias/v
':%$0 2Adam/kernel/v
:0 2Adam/bias/v
:0 2Adam/gamma/v
:0 2Adam/beta/v
':%0@ 2Adam/kernel/v
:@ 2Adam/bias/v
':%@@ 2Adam/kernel/v
:@ 2Adam/bias/v
!:
 2Adam/kernel/v
: 2Adam/bias/v
 :	@ 2Adam/kernel/v
:@ 2Adam/bias/v
:@ 2Adam/kernel/v
: 2Adam/bias/vº
"__inference__wrapped_model_1441236"4567?@OP_`wxyz°±ÈÉØÙ:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿÀð
ª "1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿµ
I__inference_activation_1_layer_call_and_return_conditional_losses_1443299h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ55$
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ55$
 Ú
I__inference_activation_1_layer_call_and_return_conditional_losses_1443304I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
 
.__inference_activation_1_layer_call_fn_1443289[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ55$
ª " ÿÿÿÿÿÿÿÿÿ55$±
.__inference_activation_1_layer_call_fn_1443294I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$µ
I__inference_activation_2_layer_call_and_return_conditional_losses_1443357h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 Ú
I__inference_activation_2_layer_call_and_return_conditional_losses_1443362I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
.__inference_activation_2_layer_call_fn_1443347[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0
ª " ÿÿÿÿÿÿÿÿÿ0±
.__inference_activation_2_layer_call_fn_1443352I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0µ
I__inference_activation_3_layer_call_and_return_conditional_losses_1443527h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

@
 Ú
I__inference_activation_3_layer_call_and_return_conditional_losses_1443532I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
.__inference_activation_3_layer_call_fn_1443517[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

@
ª " ÿÿÿÿÿÿÿÿÿ

@±
.__inference_activation_3_layer_call_fn_1443522I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@µ
I__inference_activation_4_layer_call_and_return_conditional_losses_1443585h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ú
I__inference_activation_4_layer_call_and_return_conditional_losses_1443590I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
.__inference_activation_4_layer_call_fn_1443575[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@±
.__inference_activation_4_layer_call_fn_1443580I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@§
I__inference_activation_5_layer_call_and_return_conditional_losses_1443676Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_activation_5_layer_call_fn_1443671M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
I__inference_activation_6_layer_call_and_return_conditional_losses_1443732X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
.__inference_activation_6_layer_call_fn_1443727K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@³
G__inference_activation_layer_call_and_return_conditional_losses_1443241h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿnn
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿnn
 Ø
G__inference_activation_layer_call_and_return_conditional_losses_1443246I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_activation_layer_call_fn_1443231[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿnn
ª " ÿÿÿÿÿÿÿÿÿnn¯
,__inference_activation_layer_call_fn_1443236I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1443421wxyzM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 í
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1443474wxyzM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 Å
7__inference_batch_normalization_1_layer_call_fn_1443385wxyzM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0Å
7__inference_batch_normalization_1_layer_call_fn_1443398wxyzM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0ë
P__inference_batch_normalization_layer_call_and_return_conditional_losses_14431354567M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ë
P__inference_batch_normalization_layer_call_and_return_conditional_losses_14431884567M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
5__inference_batch_normalization_layer_call_fn_14430994567M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÃ
5__inference_batch_normalization_layer_call_fn_14431124567M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1443274lOP7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿnn
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ55$
 Ú
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1443284OPI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
 
*__inference_conv2d_1_layer_call_fn_1443255_OP7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿnn
ª " ÿÿÿÿÿÿÿÿÿ55$²
*__inference_conv2d_1_layer_call_fn_1443264OPI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$µ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1443332l_`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ55$
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ0
 Ú
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1443342_`I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
 
*__inference_conv2d_2_layer_call_fn_1443313__`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ55$
ª " ÿÿÿÿÿÿÿÿÿ0²
*__inference_conv2d_2_layer_call_fn_1443322_`I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ$
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0·
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1443502n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

@
 Ü
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1443512I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_3_layer_call_fn_1443483a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ0
ª " ÿÿÿÿÿÿÿÿÿ

@´
*__inference_conv2d_3_layer_call_fn_1443492I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ0
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@·
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1443560n7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ü
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1443570I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_4_layer_call_fn_1443541a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

@
ª " ÿÿÿÿÿÿÿÿÿ@´
*__inference_conv2d_4_layer_call_fn_1443550I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@µ
C__inference_conv2d_layer_call_and_return_conditional_losses_1443216n?@9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿnn
 Ø
C__inference_conv2d_layer_call_and_return_conditional_losses_1443226?@I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
(__inference_conv2d_layer_call_fn_1443197a?@9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿàà
ª " ÿÿÿÿÿÿÿÿÿnn°
(__inference_conv2d_layer_call_fn_1443206?@I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
D__inference_dense_1_layer_call_and_return_conditional_losses_1443722_ÈÉ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_dense_1_layer_call_fn_1443712RÈÉ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¦
D__inference_dense_2_layer_call_and_return_conditional_losses_1443751^ØÙ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_2_layer_call_fn_1443741QØÙ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
B__inference_dense_layer_call_and_return_conditional_losses_1443656`°±0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ®
B__inference_dense_layer_call_and_return_conditional_losses_1443666h°±8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
'__inference_dense_layer_call_fn_1443637S°±0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_dense_layer_call_fn_1443646[°±8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dropout_layer_call_and_return_conditional_losses_1443691^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¦
D__inference_dropout_layer_call_and_return_conditional_losses_1443703^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dropout_layer_call_fn_1443681Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ~
)__inference_dropout_layer_call_fn_1443686Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ©
D__inference_flatten_layer_call_and_return_conditional_losses_1443616a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ã
D__inference_flatten_layer_call_and_return_conditional_losses_1443628{I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
)__inference_flatten_layer_call_fn_1443605T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_flatten_layer_call_fn_1443610nI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿï
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1443600R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ç
1__inference_max_pooling2d_1_layer_call_fn_1443595R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1443372R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_layer_call_fn_1443367R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÖ
B__inference_model_layer_call_and_return_conditional_losses_1442460"4567?@OP_`wxyz°±ÈÉØÙB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿÀð
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ø
B__inference_model_layer_call_and_return_conditional_losses_1442537$â4567?@OP_`wxyz°±ÈÉØÙB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿÀð
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Õ
B__inference_model_layer_call_and_return_conditional_losses_1442755"4567?@OP_`wxyz°±ÈÉØÙA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÀð
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ×
B__inference_model_layer_call_and_return_conditional_losses_1442963$â4567?@OP_`wxyz°±ÈÉØÙA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÀð
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ®
'__inference_model_layer_call_fn_1441753"4567?@OP_`wxyz°±ÈÉØÙB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿÀð
p 

 
ª "ÿÿÿÿÿÿÿÿÿ°
'__inference_model_layer_call_fn_1442385$â4567?@OP_`wxyz°±ÈÉØÙB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿÀð
p

 
ª "ÿÿÿÿÿÿÿÿÿ­
'__inference_model_layer_call_fn_1442596"4567?@OP_`wxyz°±ÈÉØÙA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÀð
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¯
'__inference_model_layer_call_fn_1442651$â4567?@OP_`wxyz°±ÈÉØÙA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿÀð
p

 
ª "ÿÿÿÿÿÿÿÿÿÀ
L__inference_random_contrast_layer_call_and_return_conditional_losses_1443045p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 Ä
L__inference_random_contrast_layer_call_and_return_conditional_losses_1443086tâ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿàà
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 
1__inference_random_contrast_layer_call_fn_1443034c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿàà
p 
ª ""ÿÿÿÿÿÿÿÿÿàà
1__inference_random_contrast_layer_call_fn_1443041gâ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿàà
p
ª ""ÿÿÿÿÿÿÿÿÿààµ
E__inference_resizing_layer_call_and_return_conditional_losses_1443029l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÀð
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿàà
 
*__inference_resizing_layer_call_fn_1443023_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿÀð
ª ""ÿÿÿÿÿÿÿÿÿààÈ
%__inference_signature_wrapper_1443018"4567?@OP_`wxyz°±ÈÉØÙE¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿÀð"1ª.
,
dense_2!
dense_2ÿÿÿÿÿÿÿÿÿ