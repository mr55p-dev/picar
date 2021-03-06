??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
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
?
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
?
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
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
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
?
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
delete_old_dirsbool(?
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
dtypetype?
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
?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
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
??*
shared_name
kernel_5
g
kernel_5/Read/ReadVariableOpReadVariableOpkernel_5* 
_output_shapes
:
??*
dtype0
e
bias_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebias_5
^
bias_5/Read/ReadVariableOpReadVariableOpbias_5*
_output_shapes	
:?*
dtype0
m
kernel_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@*
shared_name
kernel_6
f
kernel_6/Read/ReadVariableOpReadVariableOpkernel_6*
_output_shapes
:	?@*
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
?
random_contrast/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*)
shared_namerandom_contrast/StateVar
?
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
?
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
?
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
?
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
?
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
??* 
shared_nameAdam/kernel/m_5
u
#Adam/kernel/m_5/Read/ReadVariableOpReadVariableOpAdam/kernel/m_5* 
_output_shapes
:
??*
dtype0
s
Adam/bias/m_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/bias/m_5
l
!Adam/bias/m_5/Read/ReadVariableOpReadVariableOpAdam/bias/m_5*
_output_shapes	
:?*
dtype0
{
Adam/kernel/m_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_nameAdam/kernel/m_6
t
#Adam/kernel/m_6/Read/ReadVariableOpReadVariableOpAdam/kernel/m_6*
_output_shapes
:	?@*
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
?
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
?
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
?
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
?
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
??* 
shared_nameAdam/kernel/v_5
u
#Adam/kernel/v_5/Read/ReadVariableOpReadVariableOpAdam/kernel/v_5* 
_output_shapes
:
??*
dtype0
s
Adam/bias/v_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/bias/v_5
l
!Adam/bias/v_5/Read/ReadVariableOpReadVariableOpAdam/bias/v_5*
_output_shapes	
:?*
dtype0
{
Adam/kernel/v_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?@* 
shared_nameAdam/kernel/v_6
t
#Adam/kernel/v_6/Read/ReadVariableOpReadVariableOpAdam/kernel/v_6*
_output_shapes
:	?@*
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
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
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
?
#$_self_saveable_object_factories
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses* 
?
#+_self_saveable_object_factories
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses*
?
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
?

?kernel
@bias
#A_self_saveable_object_factories
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
?
#H_self_saveable_object_factories
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
?

Okernel
Pbias
#Q_self_saveable_object_factories
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses*
?
#X_self_saveable_object_factories
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
?

_kernel
`bias
#a_self_saveable_object_factories
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses*
?
#h_self_saveable_object_factories
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
?
#o_self_saveable_object_factories
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 
?
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
?__call__
+?&call_and_return_all_conditional_losses*
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses*
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate4m?5m??m?@m?Om?Pm?_m?`m?wm?xm?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?4v?5v??v?@v?Ov?Pv?_v?`v?wv?xv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*

?serving_default* 
* 
?
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
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23*
?
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
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

?
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
XR
VARIABLE_VALUEkernel_36layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_34layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEkernel_46layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_44layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEkernel_56layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_54layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
* 
XR
VARIABLE_VALUEkernel_66layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_64layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 
* 
* 
XR
VARIABLE_VALUEkernel_76layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEbias_74layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*
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
?
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
?0
?1*
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
?
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

?total

?count
?	variables
?	keras_api*
<

?total

?count
?	variables
?	keras_api*
|v
VARIABLE_VALUErandom_contrast/StateVarJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
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
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1gammabetamoving_meanmoving_variancekernelbiaskernel_1bias_1kernel_2bias_2gamma_1beta_1moving_mean_1moving_variance_1kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_7bias_7*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference_signature_wrapper_117135
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
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
GPU2*0,1J 8? *(
f#R!
__inference__traced_save_118118
?

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
GPU2*0,1J 8? *+
f&R$
"__inference__traced_restore_118350??
?
L
0__inference_max_pooling2d_1_layer_call_fn_117712

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_115582?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_117687

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
cond_false_117549
cond_placeholder
cond_placeholder_1*
cond_readvariableop_resource:0,
cond_readvariableop_1_resource:0
cond_identity_inputs
cond_identity
cond_identity_1
cond_identity_2??cond/ReadVariableOp?cond/ReadVariableOp_1?
cond/IdentityIdentitycond_identity_inputs
^cond/NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0l
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
5: : : : :+???????????????????????????02*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+???????????????????????????0
?	
?
4__inference_batch_normalization_layer_call_fn_117229

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_115446?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_115759

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_activation_1_layer_call_fn_117411

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_116192z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????$:i e
A
_output_shapes/
-:+???????????????????????????$
 
_user_specified_nameinputs
?
d
H__inference_activation_3_layer_call_and_return_conditional_losses_117649

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_4_layer_call_and_return_conditional_losses_117702

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
cond_false_115520
cond_placeholder
cond_placeholder_1*
cond_readvariableop_resource:0,
cond_readvariableop_1_resource:0
cond_identity_inputs
cond_identity
cond_identity_1
cond_identity_2??cond/ReadVariableOp?cond/ReadVariableOp_1?
cond/IdentityIdentitycond_identity_inputs
^cond/NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0l
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
5: : : : :+???????????????????????????02*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+???????????????????????????0
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_115625

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????nn*
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
:?????????nng
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????nnw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_resizing_layer_call_and_return_conditional_losses_117146

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_117538

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1;
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
valueB:?
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
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_116255

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
cond_true_117548*
cond_readvariableop_resource:0,
cond_readvariableop_1_resource:0;
-cond_fusedbatchnormv3_readvariableop_resource:0=
/cond_fusedbatchnormv3_readvariableop_1_resource:0 
cond_fusedbatchnormv3_inputs
cond_identity
cond_identity_1
cond_identity_2??$cond/FusedBatchNormV3/ReadVariableOp?&cond/FusedBatchNormV3/ReadVariableOp_1?cond/ReadVariableOp?cond/ReadVariableOp_1l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:0*
dtype0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
$cond/FusedBatchNormV3/ReadVariableOpReadVariableOp-cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
&cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
cond/FusedBatchNormV3FusedBatchNormV3cond_fusedbatchnormv3_inputscond/ReadVariableOp:value:0cond/ReadVariableOp_1:value:0,cond/FusedBatchNormV3/ReadVariableOp:value:0.cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
cond/IdentityIdentitycond/FusedBatchNormV3:y:0
^cond/NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0p
cond/Identity_1Identity"cond/FusedBatchNormV3:batch_mean:0
^cond/NoOp*
T0*
_output_shapes
:0t
cond/Identity_2Identity&cond/FusedBatchNormV3:batch_variance:0
^cond/NoOp*
T0*
_output_shapes
:0?
	cond/NoOpNoOp%^cond/FusedBatchNormV3/ReadVariableOp'^cond/FusedBatchNormV3/ReadVariableOp_1^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+???????????????????????????02L
$cond/FusedBatchNormV3/ReadVariableOp$cond/FusedBatchNormV3/ReadVariableOp2P
&cond/FusedBatchNormV3/ReadVariableOp_1&cond/FusedBatchNormV3/ReadVariableOp_12*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+???????????????????????????0
?
?
&__inference_dense_layer_call_fn_117763

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_116290p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_3_layer_call_and_return_conditional_losses_117644

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????

@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????

@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

@:W S
/
_output_shapes
:?????????

@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115704

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
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
:?????????

@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????

@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
L
0__inference_random_contrast_layer_call_fn_117151

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_random_contrast_layer_call_and_return_conditional_losses_115604j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
I
-__inference_activation_4_layer_call_fn_117697

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_116265z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
'__inference_conv2d_layer_call_fn_117323

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_116161?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_115380

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1;
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
valueB:?
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
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_115496

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1;
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
valueB:?
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
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
cond_true_117262*
cond_readvariableop_resource:,
cond_readvariableop_1_resource:;
-cond_fusedbatchnormv3_readvariableop_resource:=
/cond_fusedbatchnormv3_readvariableop_1_resource: 
cond_fusedbatchnormv3_inputs
cond_identity
cond_identity_1
cond_identity_2??$cond/FusedBatchNormV3/ReadVariableOp?&cond/FusedBatchNormV3/ReadVariableOp_1?cond/ReadVariableOp?cond/ReadVariableOp_1l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:*
dtype0?
$cond/FusedBatchNormV3/ReadVariableOpReadVariableOp-cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
&cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
cond/FusedBatchNormV3FusedBatchNormV3cond_fusedbatchnormv3_inputscond/ReadVariableOp:value:0cond/ReadVariableOp_1:value:0,cond/FusedBatchNormV3/ReadVariableOp:value:0.cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
cond/IdentityIdentitycond/FusedBatchNormV3:y:0
^cond/NoOp*
T0*A
_output_shapes/
-:+???????????????????????????p
cond/Identity_1Identity"cond/FusedBatchNormV3:batch_mean:0
^cond/NoOp*
T0*
_output_shapes
:t
cond/Identity_2Identity&cond/FusedBatchNormV3:batch_variance:0
^cond/NoOp*
T0*
_output_shapes
:?
	cond/NoOpNoOp%^cond/FusedBatchNormV3/ReadVariableOp'^cond/FusedBatchNormV3/ReadVariableOp_1^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+???????????????????????????2L
$cond/FusedBatchNormV3/ReadVariableOp$cond/FusedBatchNormV3/ReadVariableOp2P
&cond/FusedBatchNormV3/ReadVariableOp_1&cond/FusedBatchNormV3/ReadVariableOp_12*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+???????????????????????????
?
?
'batch_normalization_1_cond_false_116993*
&batch_normalization_1_cond_placeholder,
(batch_normalization_1_cond_placeholder_1@
2batch_normalization_1_cond_readvariableop_resource:0B
4batch_normalization_1_cond_readvariableop_1_resource:0=
9batch_normalization_1_cond_identity_max_pooling2d_maxpool'
#batch_normalization_1_cond_identity)
%batch_normalization_1_cond_identity_1)
%batch_normalization_1_cond_identity_2??)batch_normalization_1/cond/ReadVariableOp?+batch_normalization_1/cond/ReadVariableOp_1?
#batch_normalization_1/cond/IdentityIdentity9batch_normalization_1_cond_identity_max_pooling2d_maxpool ^batch_normalization_1/cond/NoOp*
T0*/
_output_shapes
:?????????0?
)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1_cond_readvariableop_resource*
_output_shapes
:0*
dtype0?
%batch_normalization_1/cond/Identity_1Identity1batch_normalization_1/cond/ReadVariableOp:value:0 ^batch_normalization_1/cond/NoOp*
T0*
_output_shapes
:0?
+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1_cond_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
%batch_normalization_1/cond/Identity_2Identity3batch_normalization_1/cond/ReadVariableOp_1:value:0 ^batch_normalization_1/cond/NoOp*
T0*
_output_shapes
:0?
batch_normalization_1/cond/NoOpNoOp*^batch_normalization_1/cond/ReadVariableOp,^batch_normalization_1/cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "S
#batch_normalization_1_cond_identity,batch_normalization_1/cond/Identity:output:0"W
%batch_normalization_1_cond_identity_1.batch_normalization_1/cond/Identity_1:output:0"W
%batch_normalization_1_cond_identity_2.batch_normalization_1/cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????02V
)batch_normalization_1/cond/ReadVariableOp)batch_normalization_1/cond/ReadVariableOp2Z
+batch_normalization_1/cond/ReadVariableOp_1+batch_normalization_1/cond/ReadVariableOp_1:51
/
_output_shapes
:?????????0
?/
?
K__inference_random_contrast_layer_call_and_return_conditional_losses_116070

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity??(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
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
valueB:?
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
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
valueB:?
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
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
valueB"      ?
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
 *??L?a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?????
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: |
adjust_contrastAdjustContrastv2inputsstateless_random_uniform:z:0*1
_output_shapes
:???????????z
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*1
_output_shapes
:???????????z
IdentityIdentity!adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_1_layer_call_fn_117502

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_115496?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_117459

inputs8
conv2d_readvariableop_resource:$0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????$
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_1_layer_call_fn_117515

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_115562?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_117727

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_116279i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?]
?

A__inference_model_layer_call_and_return_conditional_losses_116577
input_1(
batch_normalization_116507:(
batch_normalization_116509:(
batch_normalization_116511:(
batch_normalization_116513:'
conv2d_116516:
conv2d_116518:)
conv2d_1_116522:$
conv2d_1_116524:$)
conv2d_2_116528:$0
conv2d_2_116530:0*
batch_normalization_1_116535:0*
batch_normalization_1_116537:0*
batch_normalization_1_116539:0*
batch_normalization_1_116541:0)
conv2d_3_116544:0@
conv2d_3_116546:@)
conv2d_4_116550:@@
conv2d_4_116552:@ 
dense_116558:
??
dense_116560:	?!
dense_1_116565:	?@
dense_1_116567:@ 
dense_2_116571:@
dense_2_116573:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
resizing/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_resizing_layer_call_and_return_conditional_losses_115598?
random_contrast/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_random_contrast_layer_call_and_return_conditional_losses_115604?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(random_contrast/PartitionedCall:output:0batch_normalization_116507batch_normalization_116509batch_normalization_116511batch_normalization_116513*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_115380?
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_116516conv2d_116518*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_115625?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_115636?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_116522conv2d_1_116524*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115648?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_115659?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_116528conv2d_2_116530*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115671?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_115682?
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_115466?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_116535batch_normalization_1_116537batch_normalization_1_116539batch_normalization_1_116541*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_115496?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_3_116544conv2d_3_116546*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115704?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_115715?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_116550conv2d_4_116552*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_115727?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_115738?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_115582?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_115747?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_116558dense_116560*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_115759?
activation_5/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_115770?
dropout/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_115777?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_116565dense_1_116567*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115789?
activation_6/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_115800?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_2_116571dense_2_116573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115812w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:???????????
!
_user_specified_name	input_1
?

_
C__inference_flatten_layer_call_and_return_conditional_losses_117745

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
valueB:?
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
?????????u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
`
D__inference_resizing_layer_call_and_return_conditional_losses_115598

inputs
identity\
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
resize/ResizeBilinearResizeBilinearinputsresize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(x
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
H__inference_activation_6_layer_call_and_return_conditional_losses_117849

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
__inference__traced_save_118118
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

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*?(
value?(B?(KB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*?
value?B?KB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_gamma_read_readvariableopsavev2_beta_read_readvariableop&savev2_moving_mean_read_readvariableop*savev2_moving_variance_read_readvariableop!savev2_kernel_read_readvariableopsavev2_bias_read_readvariableop#savev2_kernel_1_read_readvariableop!savev2_bias_1_read_readvariableop#savev2_kernel_2_read_readvariableop!savev2_bias_2_read_readvariableop"savev2_gamma_1_read_readvariableop!savev2_beta_1_read_readvariableop(savev2_moving_mean_1_read_readvariableop,savev2_moving_variance_1_read_readvariableop#savev2_kernel_3_read_readvariableop!savev2_bias_3_read_readvariableop#savev2_kernel_4_read_readvariableop!savev2_bias_4_read_readvariableop#savev2_kernel_5_read_readvariableop!savev2_bias_5_read_readvariableop#savev2_kernel_6_read_readvariableop!savev2_bias_6_read_readvariableop#savev2_kernel_7_read_readvariableop!savev2_bias_7_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop3savev2_random_contrast_statevar_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop'savev2_adam_gamma_m_read_readvariableop&savev2_adam_beta_m_read_readvariableop(savev2_adam_kernel_m_read_readvariableop&savev2_adam_bias_m_read_readvariableop*savev2_adam_kernel_m_1_read_readvariableop(savev2_adam_bias_m_1_read_readvariableop*savev2_adam_kernel_m_2_read_readvariableop(savev2_adam_bias_m_2_read_readvariableop)savev2_adam_gamma_m_1_read_readvariableop(savev2_adam_beta_m_1_read_readvariableop*savev2_adam_kernel_m_3_read_readvariableop(savev2_adam_bias_m_3_read_readvariableop*savev2_adam_kernel_m_4_read_readvariableop(savev2_adam_bias_m_4_read_readvariableop*savev2_adam_kernel_m_5_read_readvariableop(savev2_adam_bias_m_5_read_readvariableop*savev2_adam_kernel_m_6_read_readvariableop(savev2_adam_bias_m_6_read_readvariableop*savev2_adam_kernel_m_7_read_readvariableop(savev2_adam_bias_m_7_read_readvariableop'savev2_adam_gamma_v_read_readvariableop&savev2_adam_beta_v_read_readvariableop(savev2_adam_kernel_v_read_readvariableop&savev2_adam_bias_v_read_readvariableop*savev2_adam_kernel_v_1_read_readvariableop(savev2_adam_bias_v_1_read_readvariableop*savev2_adam_kernel_v_2_read_readvariableop(savev2_adam_bias_v_2_read_readvariableop)savev2_adam_gamma_v_1_read_readvariableop(savev2_adam_beta_v_1_read_readvariableop*savev2_adam_kernel_v_3_read_readvariableop(savev2_adam_bias_v_3_read_readvariableop*savev2_adam_kernel_v_4_read_readvariableop(savev2_adam_bias_v_4_read_readvariableop*savev2_adam_kernel_v_5_read_readvariableop(savev2_adam_bias_v_5_read_readvariableop*savev2_adam_kernel_v_6_read_readvariableop(savev2_adam_bias_v_6_read_readvariableop*savev2_adam_kernel_v_7_read_readvariableop(savev2_adam_bias_v_7_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::$:$:$0:0:0:0:0:0:0@:@:@@:@:
??:?:	?@:@:@:: : : : : :: : : : :::::$:$:$0:0:0:0:0@:@:@@:@:
??:?:	?@:@:@::::::$:$:$0:0:0:0:0@:@:@@:@:
??:?:	?@:@:@:: 2(
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
??:!

_output_shapes	
:?:%!

_output_shapes
:	?@: 
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
??:!2

_output_shapes	
:?:%3!

_output_shapes
:	?@: 4
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
??:!F

_output_shapes	
:?:%G!

_output_shapes
:	?@: H
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
?

_
C__inference_flatten_layer_call_and_return_conditional_losses_116279

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
valueB:?
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
?????????u
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:m
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????????????a
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
I
-__inference_activation_3_layer_call_fn_117634

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_115715h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????

@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

@:W S
/
_output_shapes
:?????????

@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_2_layer_call_fn_117430

inputs!
unknown:$0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115671w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????55$: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????55$
 
_user_specified_nameinputs
?
I
-__inference_activation_2_layer_call_fn_117464

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_115682h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_layer_call_fn_117484

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_115466?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
cond_true_115403*
cond_readvariableop_resource:,
cond_readvariableop_1_resource:;
-cond_fusedbatchnormv3_readvariableop_resource:=
/cond_fusedbatchnormv3_readvariableop_1_resource: 
cond_fusedbatchnormv3_inputs
cond_identity
cond_identity_1
cond_identity_2??$cond/FusedBatchNormV3/ReadVariableOp?&cond/FusedBatchNormV3/ReadVariableOp_1?cond/ReadVariableOp?cond/ReadVariableOp_1l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:*
dtype0?
$cond/FusedBatchNormV3/ReadVariableOpReadVariableOp-cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
&cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
cond/FusedBatchNormV3FusedBatchNormV3cond_fusedbatchnormv3_inputscond/ReadVariableOp:value:0cond/ReadVariableOp_1:value:0,cond/FusedBatchNormV3/ReadVariableOp:value:0.cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
cond/IdentityIdentitycond/FusedBatchNormV3:y:0
^cond/NoOp*
T0*A
_output_shapes/
-:+???????????????????????????p
cond/Identity_1Identity"cond/FusedBatchNormV3:batch_mean:0
^cond/NoOp*
T0*
_output_shapes
:t
cond/Identity_2Identity&cond/FusedBatchNormV3:batch_variance:0
^cond/NoOp*
T0*
_output_shapes
:?
	cond/NoOpNoOp%^cond/FusedBatchNormV3/ReadVariableOp'^cond/FusedBatchNormV3/ReadVariableOp_1^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+???????????????????????????2L
$cond/FusedBatchNormV3/ReadVariableOp$cond/FusedBatchNormV3/ReadVariableOp2P
&cond/FusedBatchNormV3/ReadVariableOp_1&cond/FusedBatchNormV3/ReadVariableOp_12*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+???????????????????????????
?
d
H__inference_activation_2_layer_call_and_return_conditional_losses_115682

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_115777

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_1_layer_call_fn_117381

inputs!
unknown:$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_116182?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_5_layer_call_and_return_conditional_losses_115770

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:??????????[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_activation_4_layer_call_fn_117692

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_115738h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_117829

inputs
unknown:	?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115789o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense_2_layer_call_and_return_conditional_losses_115812

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115671

inputs8
conv2d_readvariableop_resource:$0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
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
:?????????0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????55$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????55$
 
_user_specified_nameinputs
?
I
-__inference_activation_5_layer_call_fn_117788

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_115770a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?b
?
A__inference_model_layer_call_and_return_conditional_losses_116654
input_1$
random_contrast_116581:	(
batch_normalization_116584:(
batch_normalization_116586:(
batch_normalization_116588:(
batch_normalization_116590:'
conv2d_116593:
conv2d_116595:)
conv2d_1_116599:$
conv2d_1_116601:$)
conv2d_2_116605:$0
conv2d_2_116607:0*
batch_normalization_1_116612:0*
batch_normalization_1_116614:0*
batch_normalization_1_116616:0*
batch_normalization_1_116618:0)
conv2d_3_116621:0@
conv2d_3_116623:@)
conv2d_4_116627:@@
conv2d_4_116629:@ 
dense_116635:
??
dense_116637:	?!
dense_1_116642:	?@
dense_1_116644:@ 
dense_2_116648:@
dense_2_116650:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?'random_contrast/StatefulPartitionedCall?
resizing/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_resizing_layer_call_and_return_conditional_losses_115598?
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall!resizing/PartitionedCall:output:0random_contrast_116581*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_random_contrast_layer_call_and_return_conditional_losses_116070?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0random_contrast/StatefulPartitionedCall:output:0batch_normalization_116584batch_normalization_116586batch_normalization_116588batch_normalization_116590*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_115446?
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_116593conv2d_116595*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_116161?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_116171?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_116599conv2d_1_116601*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_116182?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_116192?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_116605conv2d_2_116607*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_116203?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_116213?
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_115466?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_116612batch_normalization_1_116614batch_normalization_1_116616batch_normalization_1_116618*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_115562?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_3_116621conv2d_3_116623*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_116234?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_116244?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_116627conv2d_4_116629*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_116255?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_116265?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_115582?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_116279?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_116635dense_116637*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_116290?
activation_5/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_115770?
dropout/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_115916?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_116642dense_1_116644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115789?
activation_6/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_115800?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_2_116648dense_2_116650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115812w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall(^random_contrast/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:???????????: : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:???????????
!
_user_specified_name	input_1
?
?
0__inference_random_contrast_layer_call_fn_117158

inputs
unknown:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_random_contrast_layer_call_and_return_conditional_losses_116070y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
cond_false_117263
cond_placeholder
cond_placeholder_1*
cond_readvariableop_resource:,
cond_readvariableop_1_resource:
cond_identity_inputs
cond_identity
cond_identity_1
cond_identity_2??cond/ReadVariableOp?cond/ReadVariableOp_1?
cond/IdentityIdentitycond_identity_inputs
^cond/NoOp*
T0*A
_output_shapes/
-:+???????????????????????????l
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
5: : : : :+???????????????????????????2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+???????????????????????????
?

?
cond_false_115404
cond_placeholder
cond_placeholder_1*
cond_readvariableop_resource:,
cond_readvariableop_1_resource:
cond_identity_inputs
cond_identity
cond_identity_1
cond_identity_2??cond/ReadVariableOp?cond/ReadVariableOp_1?
cond/IdentityIdentitycond_identity_inputs
^cond/NoOp*
T0*A
_output_shapes/
-:+???????????????????????????l
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
5: : : : :+???????????????????????????2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+???????????????????????????
?
d
H__inference_activation_3_layer_call_and_return_conditional_losses_115715

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????

@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????

@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????

@:W S
/
_output_shapes
:?????????

@
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_115747

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_6_layer_call_and_return_conditional_losses_115800

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:?????????@Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_2_layer_call_and_return_conditional_losses_117474

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
)__inference_conv2d_3_layer_call_fn_117609

inputs!
unknown:0@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_116234?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_115789

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_115582

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_1_layer_call_fn_117372

inputs!
unknown:$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115648w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????55$`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????nn: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????nn
 
_user_specified_nameinputs
?
g
K__inference_random_contrast_layer_call_and_return_conditional_losses_117162

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_115562

inputs
cond_input_0:0
cond_input_1:0
cond_input_2:0
cond_input_3:0
identity??AssignNewValue?AssignNewValue_1?cond;
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
valueB:?
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
: ?
condIfGreater:z:0cond_input_0cond_input_1cond_input_2cond_input_3inputs*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*M
_output_shapes;
9:+???????????????????????????0:0:0*&
_read_only_resource_inputs
*$
else_branchR
cond_false_115520*L
output_shapes;
9:+???????????????????????????0:0:0*#
then_branchR
cond_true_115519t
cond/IdentityIdentitycond:output:0*
T0*A
_output_shapes/
-:+???????????????????????????0O
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
-:+???????????????????????????0q
NoOpNoOp^AssignNewValue^AssignNewValue_1^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12
condcond:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
I
-__inference_activation_3_layer_call_fn_117639

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_116244z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
&batch_normalization_1_cond_true_116992@
2batch_normalization_1_cond_readvariableop_resource:0B
4batch_normalization_1_cond_readvariableop_1_resource:0Q
Cbatch_normalization_1_cond_fusedbatchnormv3_readvariableop_resource:0S
Ebatch_normalization_1_cond_fusedbatchnormv3_readvariableop_1_resource:0E
Abatch_normalization_1_cond_fusedbatchnormv3_max_pooling2d_maxpool'
#batch_normalization_1_cond_identity)
%batch_normalization_1_cond_identity_1)
%batch_normalization_1_cond_identity_2??:batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp?<batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1?)batch_normalization_1/cond/ReadVariableOp?+batch_normalization_1/cond/ReadVariableOp_1?
)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1_cond_readvariableop_resource*
_output_shapes
:0*
dtype0?
+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1_cond_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
:batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOpReadVariableOpCbatch_normalization_1_cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
<batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEbatch_normalization_1_cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
+batch_normalization_1/cond/FusedBatchNormV3FusedBatchNormV3Abatch_normalization_1_cond_fusedbatchnormv3_max_pooling2d_maxpool1batch_normalization_1/cond/ReadVariableOp:value:03batch_normalization_1/cond/ReadVariableOp_1:value:0Bbatch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp:value:0Dbatch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
#batch_normalization_1/cond/IdentityIdentity/batch_normalization_1/cond/FusedBatchNormV3:y:0 ^batch_normalization_1/cond/NoOp*
T0*/
_output_shapes
:?????????0?
%batch_normalization_1/cond/Identity_1Identity8batch_normalization_1/cond/FusedBatchNormV3:batch_mean:0 ^batch_normalization_1/cond/NoOp*
T0*
_output_shapes
:0?
%batch_normalization_1/cond/Identity_2Identity<batch_normalization_1/cond/FusedBatchNormV3:batch_variance:0 ^batch_normalization_1/cond/NoOp*
T0*
_output_shapes
:0?
batch_normalization_1/cond/NoOpNoOp;^batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp=^batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1*^batch_normalization_1/cond/ReadVariableOp,^batch_normalization_1/cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "S
#batch_normalization_1_cond_identity,batch_normalization_1/cond/Identity:output:0"W
%batch_normalization_1_cond_identity_1.batch_normalization_1/cond/Identity_1:output:0"W
%batch_normalization_1_cond_identity_2.batch_normalization_1/cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#: : : : :?????????02x
:batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp:batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp2|
<batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_1<batch_normalization_1/cond/FusedBatchNormV3/ReadVariableOp_12V
)batch_normalization_1/cond/ReadVariableOp)batch_normalization_1/cond/ReadVariableOp2Z
+batch_normalization_1/cond/ReadVariableOp_1+batch_normalization_1/cond/ReadVariableOp_1:51
/
_output_shapes
:?????????0
?
?
)__inference_conv2d_4_layer_call_fn_117658

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_115727w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????

@
 
_user_specified_nameinputs
?
d
H__inference_activation_4_layer_call_and_return_conditional_losses_117707

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_117783

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_117629

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
d
H__inference_activation_5_layer_call_and_return_conditional_losses_117793

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:??????????[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_conv2d_layer_call_fn_117314

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_115625w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????nn`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_random_contrast_layer_call_and_return_conditional_losses_115604

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
G
+__inference_activation_layer_call_fn_117353

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_116171z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_dense_2_layer_call_fn_117858

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115812o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_117333

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????nn*
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
:?????????nng
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????nnw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_117733

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_117391

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????55$*
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
:?????????55$g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????55$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????nn: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????nn
 
_user_specified_nameinputs
?
d
H__inference_activation_1_layer_call_and_return_conditional_losses_117416

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????55$b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????55$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????55$:W S
/
_output_shapes
:?????????55$
 
_user_specified_nameinputs
?
b
F__inference_activation_layer_call_and_return_conditional_losses_117358

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????nnb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????nn"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????nn:W S
/
_output_shapes
:?????????nn
 
_user_specified_nameinputs
?

?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115648

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????55$*
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
:?????????55$g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????55$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????nn: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????nn
 
_user_specified_nameinputs
?

?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_115727

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
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
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????

@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_116234

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
ۉ
?
!__inference__wrapped_model_115353
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
??:
+model_dense_biasadd_readvariableop_resource:	??
,model_dense_1_matmul_readvariableop_resource:	?@;
-model_dense_1_biasadd_readvariableop_resource:@>
,model_dense_2_matmul_readvariableop_resource:@;
-model_dense_2_biasadd_readvariableop_resource:
identity??9model/batch_normalization/FusedBatchNormV3/ReadVariableOp?;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?(model/batch_normalization/ReadVariableOp?*model/batch_normalization/ReadVariableOp_1?;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?*model/batch_normalization_1/ReadVariableOp?,model/batch_normalization_1/ReadVariableOp_1?#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?%model/conv2d_3/BiasAdd/ReadVariableOp?$model/conv2d_3/Conv2D/ReadVariableOp?%model/conv2d_4/BiasAdd/ReadVariableOp?$model/conv2d_4/Conv2D/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOpk
model/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
$model/resizing/resize/ResizeBilinearResizeBilinearinput_1#model/resizing/resize/size:output:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(?
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
valueB:?
'model/batch_normalization/strided_sliceStridedSlice(model/batch_normalization/Shape:output:06model/batch_normalization/strided_slice/stack:output:08model/batch_normalization/strided_slice/stack_1:output:08model/batch_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
(model/batch_normalization/ReadVariableOpReadVariableOp1model_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0?
*model/batch_normalization/ReadVariableOp_1ReadVariableOp3model_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0?
9model/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpBmodel_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
;model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDmodel_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
*model/batch_normalization/FusedBatchNormV3FusedBatchNormV35model/resizing/resize/ResizeBilinear:resized_images:00model/batch_normalization/ReadVariableOp:value:02model/batch_normalization/ReadVariableOp_1:value:0Amodel/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Cmodel/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model/conv2d/Conv2DConv2D.model/batch_normalization/FusedBatchNormV3:y:0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????nn*
paddingVALID*
strides
?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????nnv
model/activation/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????nn?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0?
model/conv2d_1/Conv2DConv2D#model/activation/Relu:activations:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????55$*
paddingVALID*
strides
?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????55$z
model/activation_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????55$?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0?
model/conv2d_2/Conv2DConv2D%model/activation_1/Relu:activations:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingVALID*
strides
?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0z
model/activation_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0?
model/max_pooling2d/MaxPoolMaxPool%model/activation_2/Relu:activations:0*/
_output_shapes
:?????????0*
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
valueB:?
)model/batch_normalization_1/strided_sliceStridedSlice*model/batch_normalization_1/Shape:output:08model/batch_normalization_1/strided_slice/stack:output:0:model/batch_normalization_1/strided_slice/stack_1:output:0:model/batch_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
*model/batch_normalization_1/ReadVariableOpReadVariableOp3model_batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype0?
,model/batch_normalization_1/ReadVariableOp_1ReadVariableOp5model_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
;model/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpDmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
=model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFmodel_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
,model/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3$model/max_pooling2d/MaxPool:output:02model/batch_normalization_1/ReadVariableOp:value:04model/batch_normalization_1/ReadVariableOp_1:value:0Cmodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Emodel/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
is_training( ?
$model/conv2d_3/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
model/conv2d_3/Conv2DConv2D0model/batch_normalization_1/FusedBatchNormV3:y:0,model/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
paddingVALID*
strides
?
%model/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/conv2d_3/BiasAddBiasAddmodel/conv2d_3/Conv2D:output:0-model/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@z
model/activation_3/ReluRelumodel/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

@?
$model/conv2d_4/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
model/conv2d_4/Conv2DConv2D%model/activation_3/Relu:activations:0,model/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
%model/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/conv2d_4/BiasAddBiasAddmodel/conv2d_4/Conv2D:output:0-model/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@z
model/activation_4/ReluRelumodel/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
model/max_pooling2d_1/MaxPoolMaxPool%model/activation_4/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
model/flatten/ReshapeReshape&model/max_pooling2d_1/MaxPool:output:0model/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????p
model/activation_5/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????|
model/dropout/IdentityIdentity%model/activation_5/Relu:activations:0*
T0*(
_output_shapes
:???????????
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
model/dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@q
model/activation_6/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
model/dense_2/MatMulMatMul%model/activation_6/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????m
IdentityIdentitymodel/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp:^model/batch_normalization/FusedBatchNormV3/ReadVariableOp<^model/batch_normalization/FusedBatchNormV3/ReadVariableOp_1)^model/batch_normalization/ReadVariableOp+^model/batch_normalization/ReadVariableOp_1<^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp>^model/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1+^model/batch_normalization_1/ReadVariableOp-^model/batch_normalization_1/ReadVariableOp_1$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp&^model/conv2d_3/BiasAdd/ReadVariableOp%^model/conv2d_3/Conv2D/ReadVariableOp&^model/conv2d_4/BiasAdd/ReadVariableOp%^model/conv2d_4/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2v
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
:???????????
!
_user_specified_name	input_1
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_117808

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
b
F__inference_activation_layer_call_and_return_conditional_losses_116171

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_activation_layer_call_and_return_conditional_losses_117363

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_117591

inputs
cond_input_0:0
cond_input_1:0
cond_input_2:0
cond_input_3:0
identity??AssignNewValue?AssignNewValue_1?cond;
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
valueB:?
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
: ?
condIfGreater:z:0cond_input_0cond_input_1cond_input_2cond_input_3inputs*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*M
_output_shapes;
9:+???????????????????????????0:0:0*&
_read_only_resource_inputs
*$
else_branchR
cond_false_117549*L
output_shapes;
9:+???????????????????????????0:0:0*#
then_branchR
cond_true_117548t
cond/IdentityIdentitycond:output:0*
T0*A
_output_shapes/
-:+???????????????????????????0O
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
-:+???????????????????????????0q
NoOpNoOp^AssignNewValue^AssignNewValue_1^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12
condcond:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_116182

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????$*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????$y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_1_layer_call_and_return_conditional_losses_116192

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????$t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????$:i e
A
_output_shapes/
-:+???????????????????????????$
 
_user_specified_nameinputs
?

?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_117449

inputs8
conv2d_readvariableop_resource:$0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
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
:?????????0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????55$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????55$
 
_user_specified_nameinputs
?
?
)__inference_conv2d_3_layer_call_fn_117600

inputs!
unknown:0@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115704w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????

@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_117839

inputs1
matmul_readvariableop_resource:	?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_activation_2_layer_call_and_return_conditional_losses_117479

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????0t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_116203

inputs8
conv2d_readvariableop_resource:$0-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????$
 
_user_specified_nameinputs
?
d
H__inference_activation_4_layer_call_and_return_conditional_losses_116265

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_117773

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_117135
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
??

unknown_18:	?

unknown_19:	?@

unknown_20:@

unknown_21:@

unknown_22:
identity??StatefulPartitionedCall?
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
:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__wrapped_model_115353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_115916

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?/
?
K__inference_random_contrast_layer_call_and_return_conditional_losses_117203

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity??(stateful_uniform_full_int/RngReadAndSkipi
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:i
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: b
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
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
valueB:?
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
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
valueB:?
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0_
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
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
valueB"      ?
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
 *??L?a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?????
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: |
adjust_contrastAdjustContrastv2inputsstateless_random_uniform:z:0*1
_output_shapes
:???????????z
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*1
_output_shapes
:???????????z
IdentityIdentity!adjust_contrast/Identity:output:0^NoOp*
T0*1
_output_shapes
:???????????q
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_dense_layer_call_fn_117754

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_115759p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_activation_6_layer_call_fn_117844

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_115800`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????@:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_117717

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
G
+__inference_activation_layer_call_fn_117348

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_115636h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????nn"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????nn:W S
/
_output_shapes
:?????????nn
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_116290

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_4_layer_call_and_return_conditional_losses_115738

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_1_layer_call_and_return_conditional_losses_117421

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????$t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????$:i e
A
_output_shapes/
-:+???????????????????????????$
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_115870
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
??

unknown_18:	?

unknown_19:	?@

unknown_20:@

unknown_21:@

unknown_22:
identity??StatefulPartitionedCall?
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
:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_115819o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
&__inference_model_layer_call_fn_116713

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
??

unknown_18:	?

unknown_19:	?@

unknown_20:@

unknown_21:@

unknown_22:
identity??StatefulPartitionedCall?
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
:?????????*:
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_115819o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_117722

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_115747a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
A__inference_model_layer_call_and_return_conditional_losses_117080

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
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?batch_normalization/cond?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?batch_normalization_1/cond?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?8random_contrast/stateful_uniform_full_int/RngReadAndSkipe
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
resizing/resize/ResizeBilinearResizeBilinearinputsresizing/resize/size:output:0*
T0*1
_output_shapes
:???????????*
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
valueB: ?
.random_contrast/stateful_uniform_full_int/ProdProd8random_contrast/stateful_uniform_full_int/shape:output:08random_contrast/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: r
0random_contrast/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
0random_contrast/stateful_uniform_full_int/Cast_1Cast7random_contrast/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: ?
8random_contrast/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipArandom_contrast_stateful_uniform_full_int_rngreadandskip_resource9random_contrast/stateful_uniform_full_int/Cast/x:output:04random_contrast/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:?
=random_contrast/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?random_contrast/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?random_contrast/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7random_contrast/stateful_uniform_full_int/strided_sliceStridedSlice@random_contrast/stateful_uniform_full_int/RngReadAndSkip:value:0Frandom_contrast/stateful_uniform_full_int/strided_slice/stack:output:0Hrandom_contrast/stateful_uniform_full_int/strided_slice/stack_1:output:0Hrandom_contrast/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask?
1random_contrast/stateful_uniform_full_int/BitcastBitcast@random_contrast/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type0?
?random_contrast/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Arandom_contrast/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Arandom_contrast/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9random_contrast/stateful_uniform_full_int/strided_slice_1StridedSlice@random_contrast/stateful_uniform_full_int/RngReadAndSkip:value:0Hrandom_contrast/stateful_uniform_full_int/strided_slice_1/stack:output:0Jrandom_contrast/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Jrandom_contrast/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:?
3random_contrast/stateful_uniform_full_int/Bitcast_1BitcastBrandom_contrast/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0o
-random_contrast/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :?
)random_contrast/stateful_uniform_full_intStatelessRandomUniformFullIntV28random_contrast/stateful_uniform_full_int/shape:output:0<random_contrast/stateful_uniform_full_int/Bitcast_1:output:0:random_contrast/stateful_uniform_full_int/Bitcast:output:06random_contrast/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	d
random_contrast/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R ?
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
valueB"      ?
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
 *??L?q
,random_contrast/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?????
Erandom_contrast/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter&random_contrast/strided_slice:output:0* 
_output_shapes
::?
Erandom_contrast/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :?
Arandom_contrast/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV27random_contrast/stateless_random_uniform/shape:output:0Krandom_contrast/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Orandom_contrast/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Nrandom_contrast/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
: ?
,random_contrast/stateless_random_uniform/subSub5random_contrast/stateless_random_uniform/max:output:05random_contrast/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
,random_contrast/stateless_random_uniform/mulMulJrandom_contrast/stateless_random_uniform/StatelessRandomUniformV2:output:00random_contrast/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: ?
(random_contrast/stateless_random_uniformAddV20random_contrast/stateless_random_uniform/mul:z:05random_contrast/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: ?
random_contrast/adjust_contrastAdjustContrastv2/resizing/resize/ResizeBilinear:resized_images:0,random_contrast/stateless_random_uniform:z:0*1
_output_shapes
:????????????
(random_contrast/adjust_contrast/IdentityIdentity(random_contrast/adjust_contrast:output:0*
T0*1
_output_shapes
:???????????z
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
valueB:?
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
value	B : ?
batch_normalization/GreaterGreater*batch_normalization/strided_slice:output:0&batch_normalization/Greater/y:output:0*
T0*
_output_shapes
: ?
batch_normalization/condIfbatch_normalization/Greater:z:0 batch_normalization_cond_input_0 batch_normalization_cond_input_1 batch_normalization_cond_input_2 batch_normalization_cond_input_31random_contrast/adjust_contrast/Identity:output:0*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*=
_output_shapes+
):???????????::*&
_read_only_resource_inputs
*8
else_branch)R'
%batch_normalization_cond_false_116922*<
output_shapes+
):???????????::*7
then_branch(R&
$batch_normalization_cond_true_116921?
!batch_normalization/cond/IdentityIdentity!batch_normalization/cond:output:0*
T0*1
_output_shapes
:???????????w
#batch_normalization/cond/Identity_1Identity!batch_normalization/cond:output:1*
T0*
_output_shapes
:w
#batch_normalization/cond/Identity_2Identity!batch_normalization/cond:output:2*
T0*
_output_shapes
:?
"batch_normalization/AssignNewValueAssignVariableOp batch_normalization_cond_input_2,batch_normalization/cond/Identity_1:output:0^batch_normalization/cond*
_output_shapes
 *
dtype0?
$batch_normalization/AssignNewValue_1AssignVariableOp batch_normalization_cond_input_3,batch_normalization/cond/Identity_2:output:0^batch_normalization/cond*
_output_shapes
 *
dtype0?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2D*batch_normalization/cond/Identity:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????nn*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????nnj
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????nn?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????55$*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????55$n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????55$?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0?
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0?
max_pooling2d/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:?????????0*
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
valueB:?
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
value	B : ?
batch_normalization_1/GreaterGreater,batch_normalization_1/strided_slice:output:0(batch_normalization_1/Greater/y:output:0*
T0*
_output_shapes
: ?
batch_normalization_1/condIf!batch_normalization_1/Greater:z:0"batch_normalization_1_cond_input_0"batch_normalization_1_cond_input_1"batch_normalization_1_cond_input_2"batch_normalization_1_cond_input_3max_pooling2d/MaxPool:output:0*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*;
_output_shapes)
':?????????0:0:0*&
_read_only_resource_inputs
*:
else_branch+R)
'batch_normalization_1_cond_false_116993*:
output_shapes)
':?????????0:0:0*9
then_branch*R(
&batch_normalization_1_cond_true_116992?
#batch_normalization_1/cond/IdentityIdentity#batch_normalization_1/cond:output:0*
T0*/
_output_shapes
:?????????0{
%batch_normalization_1/cond/Identity_1Identity#batch_normalization_1/cond:output:1*
T0*
_output_shapes
:0{
%batch_normalization_1/cond/Identity_2Identity#batch_normalization_1/cond:output:2*
T0*
_output_shapes
:0?
$batch_normalization_1/AssignNewValueAssignVariableOp"batch_normalization_1_cond_input_2.batch_normalization_1/cond/Identity_1:output:0^batch_normalization_1/cond*
_output_shapes
 *
dtype0?
&batch_normalization_1/AssignNewValue_1AssignVariableOp"batch_normalization_1_cond_input_3.batch_normalization_1/cond/Identity_2:output:0^batch_normalization_1/cond*
_output_shapes
 *
dtype0?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
conv2d_3/Conv2DConv2D,batch_normalization_1/cond/Identity:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

@?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@n
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_1/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
activation_5/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
dropout/dropout/MulMulactivation_5/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????d
dropout/dropout/ShapeShapeactivation_5/Relu:activations:0*
T0*
_output_shapes
:?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@e
activation_6/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_2/MatMulMatMulactivation_6/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1^batch_normalization/cond%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1^batch_normalization_1/cond^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp9^random_contrast/stateful_uniform_full_int/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:???????????: : : : : : : : : : : : : : : : : : : : : : : : : 2H
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
:???????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_layer_call_fn_117216

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_115380?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_116502
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
??

unknown_19:	?

unknown_20:	?@

unknown_21:@

unknown_22:@

unknown_23:
identity??StatefulPartitionedCall?
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
:?????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_116310o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:???????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
%batch_normalization_cond_false_116922(
$batch_normalization_cond_placeholder*
&batch_normalization_cond_placeholder_1>
0batch_normalization_cond_readvariableop_resource:@
2batch_normalization_cond_readvariableop_1_resource:N
Jbatch_normalization_cond_identity_random_contrast_adjust_contrast_identity%
!batch_normalization_cond_identity'
#batch_normalization_cond_identity_1'
#batch_normalization_cond_identity_2??'batch_normalization/cond/ReadVariableOp?)batch_normalization/cond/ReadVariableOp_1?
!batch_normalization/cond/IdentityIdentityJbatch_normalization_cond_identity_random_contrast_adjust_contrast_identity^batch_normalization/cond/NoOp*
T0*1
_output_shapes
:????????????
'batch_normalization/cond/ReadVariableOpReadVariableOp0batch_normalization_cond_readvariableop_resource*
_output_shapes
:*
dtype0?
#batch_normalization/cond/Identity_1Identity/batch_normalization/cond/ReadVariableOp:value:0^batch_normalization/cond/NoOp*
T0*
_output_shapes
:?
)batch_normalization/cond/ReadVariableOp_1ReadVariableOp2batch_normalization_cond_readvariableop_1_resource*
_output_shapes
:*
dtype0?
#batch_normalization/cond/Identity_2Identity1batch_normalization/cond/ReadVariableOp_1:value:0^batch_normalization/cond/NoOp*
T0*
_output_shapes
:?
batch_normalization/cond/NoOpNoOp(^batch_normalization/cond/ReadVariableOp*^batch_normalization/cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "O
!batch_normalization_cond_identity*batch_normalization/cond/Identity:output:0"S
#batch_normalization_cond_identity_1,batch_normalization/cond/Identity_1:output:0"S
#batch_normalization_cond_identity_2,batch_normalization/cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :???????????2R
'batch_normalization/cond/ReadVariableOp'batch_normalization/cond/ReadVariableOp2V
)batch_normalization/cond/ReadVariableOp_1)batch_normalization/cond/ReadVariableOp_1:73
1
_output_shapes
:???????????
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_117305

inputs
cond_input_0:
cond_input_1:
cond_input_2:
cond_input_3:
identity??AssignNewValue?AssignNewValue_1?cond;
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
valueB:?
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
: ?
condIfGreater:z:0cond_input_0cond_input_1cond_input_2cond_input_3inputs*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*M
_output_shapes;
9:+???????????????????????????::*&
_read_only_resource_inputs
*$
else_branchR
cond_false_117263*L
output_shapes;
9:+???????????????????????????::*#
then_branchR
cond_true_117262t
cond/IdentityIdentitycond:output:0*
T0*A
_output_shapes/
-:+???????????????????????????O
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
-:+???????????????????????????q
NoOpNoOp^AssignNewValue^AssignNewValue_1^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12
condcond:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_1_layer_call_and_return_conditional_losses_115659

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????55$b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????55$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????55$:W S
/
_output_shapes
:?????????55$
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_115446

inputs
cond_input_0:
cond_input_1:
cond_input_2:
cond_input_3:
identity??AssignNewValue?AssignNewValue_1?cond;
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
valueB:?
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
: ?
condIfGreater:z:0cond_input_0cond_input_1cond_input_2cond_input_3inputs*
Tcond0
*
Tin	
2*
Tout
2*
_lower_using_switch_merge(*M
_output_shapes;
9:+???????????????????????????::*&
_read_only_resource_inputs
*$
else_branchR
cond_false_115404*L
output_shapes;
9:+???????????????????????????::*#
then_branchR
cond_true_115403t
cond/IdentityIdentitycond:output:0*
T0*A
_output_shapes/
-:+???????????????????????????O
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
-:+???????????????????????????q
NoOpNoOp^AssignNewValue^AssignNewValue_1^cond*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12
condcond:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
$batch_normalization_cond_true_116921>
0batch_normalization_cond_readvariableop_resource:@
2batch_normalization_cond_readvariableop_1_resource:O
Abatch_normalization_cond_fusedbatchnormv3_readvariableop_resource:Q
Cbatch_normalization_cond_fusedbatchnormv3_readvariableop_1_resource:V
Rbatch_normalization_cond_fusedbatchnormv3_random_contrast_adjust_contrast_identity%
!batch_normalization_cond_identity'
#batch_normalization_cond_identity_1'
#batch_normalization_cond_identity_2??8batch_normalization/cond/FusedBatchNormV3/ReadVariableOp?:batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1?'batch_normalization/cond/ReadVariableOp?)batch_normalization/cond/ReadVariableOp_1?
'batch_normalization/cond/ReadVariableOpReadVariableOp0batch_normalization_cond_readvariableop_resource*
_output_shapes
:*
dtype0?
)batch_normalization/cond/ReadVariableOp_1ReadVariableOp2batch_normalization_cond_readvariableop_1_resource*
_output_shapes
:*
dtype0?
8batch_normalization/cond/FusedBatchNormV3/ReadVariableOpReadVariableOpAbatch_normalization_cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
:batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCbatch_normalization_cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
)batch_normalization/cond/FusedBatchNormV3FusedBatchNormV3Rbatch_normalization_cond_fusedbatchnormv3_random_contrast_adjust_contrast_identity/batch_normalization/cond/ReadVariableOp:value:01batch_normalization/cond/ReadVariableOp_1:value:0@batch_normalization/cond/FusedBatchNormV3/ReadVariableOp:value:0Bbatch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
!batch_normalization/cond/IdentityIdentity-batch_normalization/cond/FusedBatchNormV3:y:0^batch_normalization/cond/NoOp*
T0*1
_output_shapes
:????????????
#batch_normalization/cond/Identity_1Identity6batch_normalization/cond/FusedBatchNormV3:batch_mean:0^batch_normalization/cond/NoOp*
T0*
_output_shapes
:?
#batch_normalization/cond/Identity_2Identity:batch_normalization/cond/FusedBatchNormV3:batch_variance:0^batch_normalization/cond/NoOp*
T0*
_output_shapes
:?
batch_normalization/cond/NoOpNoOp9^batch_normalization/cond/FusedBatchNormV3/ReadVariableOp;^batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1(^batch_normalization/cond/ReadVariableOp*^batch_normalization/cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "O
!batch_normalization_cond_identity*batch_normalization/cond/Identity:output:0"S
#batch_normalization_cond_identity_1,batch_normalization/cond/Identity_1:output:0"S
#batch_normalization_cond_identity_2,batch_normalization/cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :???????????2t
8batch_normalization/cond/FusedBatchNormV3/ReadVariableOp8batch_normalization/cond/FusedBatchNormV3/ReadVariableOp2x
:batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_1:batch_normalization/cond/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization/cond/ReadVariableOp'batch_normalization/cond/ReadVariableOp2V
)batch_normalization/cond/ReadVariableOp_1)batch_normalization/cond/ReadVariableOp_1:73
1
_output_shapes
:???????????
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_117252

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1;
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
valueB:?
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
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_117803

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_115916p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_activation_1_layer_call_fn_117406

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_115659h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????55$"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????55$:W S
/
_output_shapes
:?????????55$
 
_user_specified_nameinputs
?
d
H__inference_activation_2_layer_call_and_return_conditional_losses_116213

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????0t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?	
?
C__inference_dense_2_layer_call_and_return_conditional_losses_117868

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_activation_2_layer_call_fn_117469

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_116213z
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????0:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_117343

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_115466

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_2_layer_call_fn_117439

inputs!
unknown:$0
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_116203?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????$
 
_user_specified_nameinputs
??
?)
"__inference__traced_restore_118350
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
??)
assignvariableop_19_bias_5:	?/
assignvariableop_20_kernel_6:	?@(
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
??0
!assignvariableop_49_adam_bias_m_5:	?6
#assignvariableop_50_adam_kernel_m_6:	?@/
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
??0
!assignvariableop_69_adam_bias_v_5:	?6
#assignvariableop_70_adam_kernel_v_6:	?@/
!assignvariableop_71_adam_bias_v_6:@5
#assignvariableop_72_adam_kernel_v_7:@/
!assignvariableop_73_adam_bias_v_7:
identity_75??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_8?AssignVariableOp_9?)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*?(
value?(B?(KB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBJlayer-2/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:K*
dtype0*?
value?B?KB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_kernel_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_bias_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_kernel_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_bias_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_gamma_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_moving_mean_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp%assignvariableop_13_moving_variance_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_kernel_3Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_bias_3Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_kernel_4Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_bias_4Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_kernel_5Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_bias_5Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_kernel_6Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_bias_6Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_kernel_7Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_bias_7Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_random_contrast_statevarIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_totalIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_countIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp assignvariableop_34_adam_gamma_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_beta_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp!assignvariableop_36_adam_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpassignvariableop_37_adam_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp#assignvariableop_38_adam_kernel_m_1Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp!assignvariableop_39_adam_bias_m_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp#assignvariableop_40_adam_kernel_m_2Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp!assignvariableop_41_adam_bias_m_2Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp"assignvariableop_42_adam_gamma_m_1Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp!assignvariableop_43_adam_beta_m_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp#assignvariableop_44_adam_kernel_m_3Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp!assignvariableop_45_adam_bias_m_3Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp#assignvariableop_46_adam_kernel_m_4Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp!assignvariableop_47_adam_bias_m_4Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp#assignvariableop_48_adam_kernel_m_5Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp!assignvariableop_49_adam_bias_m_5Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp#assignvariableop_50_adam_kernel_m_6Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp!assignvariableop_51_adam_bias_m_6Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp#assignvariableop_52_adam_kernel_m_7Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp!assignvariableop_53_adam_bias_m_7Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp assignvariableop_54_adam_gamma_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp!assignvariableop_56_adam_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp#assignvariableop_58_adam_kernel_v_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp!assignvariableop_59_adam_bias_v_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp#assignvariableop_60_adam_kernel_v_2Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp!assignvariableop_61_adam_bias_v_2Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp"assignvariableop_62_adam_gamma_v_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp!assignvariableop_63_adam_beta_v_1Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp#assignvariableop_64_adam_kernel_v_3Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp!assignvariableop_65_adam_bias_v_3Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp#assignvariableop_66_adam_kernel_v_4Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp!assignvariableop_67_adam_bias_v_4Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp#assignvariableop_68_adam_kernel_v_5Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp!assignvariableop_69_adam_bias_v_5Identity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp#assignvariableop_70_adam_kernel_v_6Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp!assignvariableop_71_adam_bias_v_6Identity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp#assignvariableop_72_adam_kernel_v_7Identity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp!assignvariableop_73_adam_bias_v_7Identity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_74Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_75IdentityIdentity_74:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_75Identity_75:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
?

?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_117677

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
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
:?????????@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????

@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????

@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_4_layer_call_fn_117667

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_116255?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?]
?

A__inference_model_layer_call_and_return_conditional_losses_115819

inputs(
batch_normalization_115606:(
batch_normalization_115608:(
batch_normalization_115610:(
batch_normalization_115612:'
conv2d_115626:
conv2d_115628:)
conv2d_1_115649:$
conv2d_1_115651:$)
conv2d_2_115672:$0
conv2d_2_115674:0*
batch_normalization_1_115685:0*
batch_normalization_1_115687:0*
batch_normalization_1_115689:0*
batch_normalization_1_115691:0)
conv2d_3_115705:0@
conv2d_3_115707:@)
conv2d_4_115728:@@
conv2d_4_115730:@ 
dense_115760:
??
dense_115762:	?!
dense_1_115790:	?@
dense_1_115792:@ 
dense_2_115813:@
dense_2_115815:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
resizing/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_resizing_layer_call_and_return_conditional_losses_115598?
random_contrast/PartitionedCallPartitionedCall!resizing/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_random_contrast_layer_call_and_return_conditional_losses_115604?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall(random_contrast/PartitionedCall:output:0batch_normalization_115606batch_normalization_115608batch_normalization_115610batch_normalization_115612*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_115380?
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_115626conv2d_115628*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_115625?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_115636?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_115649conv2d_1_115651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115648?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????55$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_115659?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_115672conv2d_2_115674*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115671?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_115682?
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_115466?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_115685batch_normalization_1_115687batch_normalization_1_115689batch_normalization_1_115691*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_115496?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_3_115705conv2d_3_115707*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115704?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_115715?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_115728conv2d_4_115730*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_115727?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_115738?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_115582?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_115747?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_115760dense_115762*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_115759?
activation_5/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_115770?
dropout/PartitionedCallPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_115777?
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_115790dense_1_115792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115789?
activation_6/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_115800?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_2_115813dense_2_115815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115812w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:???????????
 
_user_specified_nameinputs
?
d
H__inference_activation_3_layer_call_and_return_conditional_losses_116244

inputs
identity`
ReluReluinputs*
T0*A
_output_shapes/
-:+???????????????????????????@t
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_117401

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????$*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????$y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????$w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
cond_true_115519*
cond_readvariableop_resource:0,
cond_readvariableop_1_resource:0;
-cond_fusedbatchnormv3_readvariableop_resource:0=
/cond_fusedbatchnormv3_readvariableop_1_resource:0 
cond_fusedbatchnormv3_inputs
cond_identity
cond_identity_1
cond_identity_2??$cond/FusedBatchNormV3/ReadVariableOp?&cond/FusedBatchNormV3/ReadVariableOp_1?cond/ReadVariableOp?cond/ReadVariableOp_1l
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:0*
dtype0p
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
$cond/FusedBatchNormV3/ReadVariableOpReadVariableOp-cond_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
&cond/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/cond_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
cond/FusedBatchNormV3FusedBatchNormV3cond_fusedbatchnormv3_inputscond/ReadVariableOp:value:0cond/ReadVariableOp_1:value:0,cond/FusedBatchNormV3/ReadVariableOp:value:0.cond/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????0:0:0:0:0:*
epsilon%o?:*
exponential_avg_factor%
?#<?
cond/IdentityIdentitycond/FusedBatchNormV3:y:0
^cond/NoOp*
T0*A
_output_shapes/
-:+???????????????????????????0p
cond/Identity_1Identity"cond/FusedBatchNormV3:batch_mean:0
^cond/NoOp*
T0*
_output_shapes
:0t
cond/Identity_2Identity&cond/FusedBatchNormV3:batch_variance:0
^cond/NoOp*
T0*
_output_shapes
:0?
	cond/NoOpNoOp%^cond/FusedBatchNormV3/ReadVariableOp'^cond/FusedBatchNormV3/ReadVariableOp_1^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "'
cond_identitycond/Identity:output:0"+
cond_identity_1cond/Identity_1:output:0"+
cond_identity_2cond/Identity_2:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5: : : : :+???????????????????????????02L
$cond/FusedBatchNormV3/ReadVariableOp$cond/FusedBatchNormV3/ReadVariableOp2P
&cond/FusedBatchNormV3/ReadVariableOp_1&cond/FusedBatchNormV3/ReadVariableOp_12*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:GC
A
_output_shapes/
-:+???????????????????????????0
?

?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_117619

inputs8
conv2d_readvariableop_resource:0@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
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
:?????????

@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????

@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
E
)__inference_resizing_layer_call_fn_117140

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_resizing_layer_call_and_return_conditional_losses_115598j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_117489

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
b
C__inference_dropout_layer_call_and_return_conditional_losses_117820

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_116768

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
??

unknown_19:	?

unknown_20:	?@

unknown_21:@

unknown_22:@

unknown_23:
identity??StatefulPartitionedCall?
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
:?????????*6
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_116310o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:???????????: : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_116161

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_117798

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_115777a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?b
?
A__inference_model_layer_call_and_return_conditional_losses_116310

inputs$
random_contrast_116140:	(
batch_normalization_116143:(
batch_normalization_116145:(
batch_normalization_116147:(
batch_normalization_116149:'
conv2d_116162:
conv2d_116164:)
conv2d_1_116183:$
conv2d_1_116185:$)
conv2d_2_116204:$0
conv2d_2_116206:0*
batch_normalization_1_116216:0*
batch_normalization_1_116218:0*
batch_normalization_1_116220:0*
batch_normalization_1_116222:0)
conv2d_3_116235:0@
conv2d_3_116237:@)
conv2d_4_116256:@@
conv2d_4_116258:@ 
dense_116291:
??
dense_116293:	?!
dense_1_116298:	?@
dense_1_116300:@ 
dense_2_116304:@
dense_2_116306:
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?'random_contrast/StatefulPartitionedCall?
resizing/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_resizing_layer_call_and_return_conditional_losses_115598?
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall!resizing/PartitionedCall:output:0random_contrast_116140*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_random_contrast_layer_call_and_return_conditional_losses_116070?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall0random_contrast/StatefulPartitionedCall:output:0batch_normalization_116143batch_normalization_116145batch_normalization_116147batch_normalization_116149*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_115446?
conv2d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_116162conv2d_116164*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_116161?
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_activation_layer_call_and_return_conditional_losses_116171?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_1_116183conv2d_1_116185*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_116182?
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_1_layer_call_and_return_conditional_losses_116192?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_2_116204conv2d_2_116206*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_116203?
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_2_layer_call_and_return_conditional_losses_116213?
max_pooling2d/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_115466?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_1_116216batch_normalization_1_116218batch_normalization_1_116220batch_normalization_1_116222*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_115562?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_3_116235conv2d_3_116237*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_116234?
activation_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_3_layer_call_and_return_conditional_losses_116244?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0conv2d_4_116256conv2d_4_116258*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_116255?
activation_4/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_4_layer_call_and_return_conditional_losses_116265?
max_pooling2d_1/PartitionedCallPartitionedCall%activation_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_115582?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_116279?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_116291dense_116293*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_116290?
activation_5/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_5_layer_call_and_return_conditional_losses_115770?
dropout/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_115916?
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_116298dense_1_116300*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115789?
activation_6/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_activation_6_layer_call_and_return_conditional_losses_115800?
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0dense_2_116304dense_2_116306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115812w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall(^random_contrast/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:???????????: : : : : : : : : : : : : : : : : : : : : : : : : 2Z
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
:???????????
 
_user_specified_nameinputs
?
b
F__inference_activation_layer_call_and_return_conditional_losses_115636

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????nnb
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????nn"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????nn:W S
/
_output_shapes
:?????????nn
 
_user_specified_nameinputs
?|
?
A__inference_model_layer_call_and_return_conditional_losses_116872

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
??4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@5
'dense_2_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpe
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"?   ?   ?
resizing/resize/ResizeBilinearResizeBilinearinputsresizing/resize/size:output:0*
T0*1
_output_shapes
:???????????*
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
valueB:?
!batch_normalization/strided_sliceStridedSlice"batch_normalization/Shape:output:00batch_normalization/strided_slice/stack:output:02batch_normalization/strided_slice/stack_1:output:02batch_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3/resizing/resize/ResizeBilinear:resized_images:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????nn*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????nnj
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????nn?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype0?
conv2d_1/Conv2DConv2Dactivation/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????55$*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????55$n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????55$?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:$0*
dtype0?
conv2d_2/Conv2DConv2Dactivation_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0*
paddingVALID*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0?
max_pooling2d/MaxPoolMaxPoolactivation_2/Relu:activations:0*/
_output_shapes
:?????????0*
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
valueB:?
#batch_normalization_1/strided_sliceStridedSlice$batch_normalization_1/Shape:output:02batch_normalization_1/strided_slice/stack:output:04batch_normalization_1/strided_slice/stack_1:output:04batch_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:0*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????0:0:0:0:0:*
epsilon%o?:*
is_training( ?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:0@*
dtype0?
conv2d_3/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
paddingVALID*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@n
activation_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

@?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_4/Conv2DConv2Dactivation_3/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@n
activation_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_1/MaxPoolMaxPoolactivation_4/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????d
activation_5/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????p
dropout/IdentityIdentityactivation_5/Relu:activations:0*
T0*(
_output_shapes
:???????????
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?@*
dtype0?
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@e
activation_6/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_2/MatMulMatMulactivation_6/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:???????????: : : : : : : : : : : : : : : : : : : : : : : : 2j
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
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ӭ
?
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
?
#$_self_saveable_object_factories
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#+_self_saveable_object_factories
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0_random_generator
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
?
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
?

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
?
#H_self_saveable_object_factories
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
?

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
?
#X_self_saveable_object_factories
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
?

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
?
#h_self_saveable_object_factories
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#o_self_saveable_object_factories
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
?
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
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?_random_generator
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
$?_self_saveable_object_factories
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate4m?5m??m?@m?Om?Pm?_m?`m?wm?xm?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?4v?5v??v?@v?Ov?Pv?_v?`v?wv?xv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
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
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23"
trackable_list_wrapper
?
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
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_model_layer_call_fn_115870
&__inference_model_layer_call_fn_116713
&__inference_model_layer_call_fn_116768
&__inference_model_layer_call_fn_116502?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_layer_call_and_return_conditional_losses_116872
A__inference_model_layer_call_and_return_conditional_losses_117080
A__inference_model_layer_call_and_return_conditional_losses_116577
A__inference_model_layer_call_and_return_conditional_losses_116654?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_115353input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_resizing_layer_call_fn_117140?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_resizing_layer_call_and_return_conditional_losses_117146?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
,	variables
-trainable_variables
.regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
/
?
_generator"
_generic_user_object
?2?
0__inference_random_contrast_layer_call_fn_117151
0__inference_random_contrast_layer_call_fn_117158?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_random_contrast_layer_call_and_return_conditional_losses_117162
K__inference_random_contrast_layer_call_and_return_conditional_losses_117203?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_batch_normalization_layer_call_fn_117216
4__inference_batch_normalization_layer_call_fn_117229?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_117252
O__inference_batch_normalization_layer_call_and_return_conditional_losses_117305?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_conv2d_layer_call_fn_117314
'__inference_conv2d_layer_call_fn_117323?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_layer_call_and_return_conditional_losses_117333
B__inference_conv2d_layer_call_and_return_conditional_losses_117343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
?2?
+__inference_activation_layer_call_fn_117348
+__inference_activation_layer_call_fn_117353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_activation_layer_call_and_return_conditional_losses_117358
F__inference_activation_layer_call_and_return_conditional_losses_117363?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_conv2d_1_layer_call_fn_117372
)__inference_conv2d_1_layer_call_fn_117381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_117391
D__inference_conv2d_1_layer_call_and_return_conditional_losses_117401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_activation_1_layer_call_fn_117406
-__inference_activation_1_layer_call_fn_117411?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_1_layer_call_and_return_conditional_losses_117416
H__inference_activation_1_layer_call_and_return_conditional_losses_117421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_conv2d_2_layer_call_fn_117430
)__inference_conv2d_2_layer_call_fn_117439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_117449
D__inference_conv2d_2_layer_call_and_return_conditional_losses_117459?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_activation_2_layer_call_fn_117464
-__inference_activation_2_layer_call_fn_117469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_2_layer_call_and_return_conditional_losses_117474
H__inference_activation_2_layer_call_and_return_conditional_losses_117479?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_max_pooling2d_layer_call_fn_117484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_117489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
6__inference_batch_normalization_1_layer_call_fn_117502
6__inference_batch_normalization_1_layer_call_fn_117515?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_117538
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_117591?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
": 0@ 2kernel
:@ 2bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_conv2d_3_layer_call_fn_117600
)__inference_conv2d_3_layer_call_fn_117609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_117619
D__inference_conv2d_3_layer_call_and_return_conditional_losses_117629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_activation_3_layer_call_fn_117634
-__inference_activation_3_layer_call_fn_117639?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_3_layer_call_and_return_conditional_losses_117644
H__inference_activation_3_layer_call_and_return_conditional_losses_117649?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
": @@ 2kernel
:@ 2bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_conv2d_4_layer_call_fn_117658
)__inference_conv2d_4_layer_call_fn_117667?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_117677
D__inference_conv2d_4_layer_call_and_return_conditional_losses_117687?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_activation_4_layer_call_fn_117692
-__inference_activation_4_layer_call_fn_117697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_4_layer_call_and_return_conditional_losses_117702
H__inference_activation_4_layer_call_and_return_conditional_losses_117707?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_max_pooling2d_1_layer_call_fn_117712?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_117717?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_flatten_layer_call_fn_117722
(__inference_flatten_layer_call_fn_117727?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_117733
C__inference_flatten_layer_call_and_return_conditional_losses_117745?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:
?? 2kernel
:? 2bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
&__inference_dense_layer_call_fn_117754
&__inference_dense_layer_call_fn_117763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_117773
A__inference_dense_layer_call_and_return_conditional_losses_117783?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_activation_5_layer_call_fn_117788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_5_layer_call_and_return_conditional_losses_117793?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
(__inference_dropout_layer_call_fn_117798
(__inference_dropout_layer_call_fn_117803?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_117808
C__inference_dropout_layer_call_and_return_conditional_losses_117820?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	?@ 2kernel
:@ 2bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_dense_1_layer_call_fn_117829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_117839?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_activation_6_layer_call_fn_117844?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_activation_6_layer_call_and_return_conditional_losses_117849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:@ 2kernel
: 2bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_dense_2_layer_call_fn_117858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_2_layer_call_and_return_conditional_losses_117868?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	  (2	Adam/iter
:  (2Adam/beta_1
:  (2Adam/beta_2
:  (2
Adam/decay
:  (2Adam/learning_rate
?B?
$__inference_signature_wrapper_117135input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
<
60
71
y2
z3"
trackable_list_wrapper
?
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
?0
?1"
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
?
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

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
&:$	 2random_contrast/StateVar
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
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
?? 2Adam/kernel/m
:? 2Adam/bias/m
 :	?@ 2Adam/kernel/m
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
?? 2Adam/kernel/v
:? 2Adam/bias/v
 :	?@ 2Adam/kernel/v
:@ 2Adam/bias/v
:@ 2Adam/kernel/v
: 2Adam/bias/v?
!__inference__wrapped_model_115353?"4567?@OP_`wxyz??????????:?7
0?-
+?(
input_1???????????
? "1?.
,
dense_2!?
dense_2??????????
H__inference_activation_1_layer_call_and_return_conditional_losses_117416h7?4
-?*
(?%
inputs?????????55$
? "-?*
#? 
0?????????55$
? ?
H__inference_activation_1_layer_call_and_return_conditional_losses_117421?I?F
??<
:?7
inputs+???????????????????????????$
? "??<
5?2
0+???????????????????????????$
? ?
-__inference_activation_1_layer_call_fn_117406[7?4
-?*
(?%
inputs?????????55$
? " ??????????55$?
-__inference_activation_1_layer_call_fn_117411I?F
??<
:?7
inputs+???????????????????????????$
? "2?/+???????????????????????????$?
H__inference_activation_2_layer_call_and_return_conditional_losses_117474h7?4
-?*
(?%
inputs?????????0
? "-?*
#? 
0?????????0
? ?
H__inference_activation_2_layer_call_and_return_conditional_losses_117479?I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+???????????????????????????0
? ?
-__inference_activation_2_layer_call_fn_117464[7?4
-?*
(?%
inputs?????????0
? " ??????????0?
-__inference_activation_2_layer_call_fn_117469I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+???????????????????????????0?
H__inference_activation_3_layer_call_and_return_conditional_losses_117644h7?4
-?*
(?%
inputs?????????

@
? "-?*
#? 
0?????????

@
? ?
H__inference_activation_3_layer_call_and_return_conditional_losses_117649?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
-__inference_activation_3_layer_call_fn_117634[7?4
-?*
(?%
inputs?????????

@
? " ??????????

@?
-__inference_activation_3_layer_call_fn_117639I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
H__inference_activation_4_layer_call_and_return_conditional_losses_117702h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
H__inference_activation_4_layer_call_and_return_conditional_losses_117707?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
-__inference_activation_4_layer_call_fn_117692[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
-__inference_activation_4_layer_call_fn_117697I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
H__inference_activation_5_layer_call_and_return_conditional_losses_117793Z0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ~
-__inference_activation_5_layer_call_fn_117788M0?-
&?#
!?
inputs??????????
? "????????????
H__inference_activation_6_layer_call_and_return_conditional_losses_117849X/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? |
-__inference_activation_6_layer_call_fn_117844K/?,
%?"
 ?
inputs?????????@
? "??????????@?
F__inference_activation_layer_call_and_return_conditional_losses_117358h7?4
-?*
(?%
inputs?????????nn
? "-?*
#? 
0?????????nn
? ?
F__inference_activation_layer_call_and_return_conditional_losses_117363?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_activation_layer_call_fn_117348[7?4
-?*
(?%
inputs?????????nn
? " ??????????nn?
+__inference_activation_layer_call_fn_117353I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_117538?wxyzM?J
C?@
:?7
inputs+???????????????????????????0
p 
? "??<
5?2
0+???????????????????????????0
? ?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_117591?wxyzM?J
C?@
:?7
inputs+???????????????????????????0
p
? "??<
5?2
0+???????????????????????????0
? ?
6__inference_batch_normalization_1_layer_call_fn_117502?wxyzM?J
C?@
:?7
inputs+???????????????????????????0
p 
? "2?/+???????????????????????????0?
6__inference_batch_normalization_1_layer_call_fn_117515?wxyzM?J
C?@
:?7
inputs+???????????????????????????0
p
? "2?/+???????????????????????????0?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_117252?4567M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_117305?4567M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
4__inference_batch_normalization_layer_call_fn_117216?4567M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_117229?4567M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
D__inference_conv2d_1_layer_call_and_return_conditional_losses_117391lOP7?4
-?*
(?%
inputs?????????nn
? "-?*
#? 
0?????????55$
? ?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_117401?OPI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????$
? ?
)__inference_conv2d_1_layer_call_fn_117372_OP7?4
-?*
(?%
inputs?????????nn
? " ??????????55$?
)__inference_conv2d_1_layer_call_fn_117381?OPI?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????$?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_117449l_`7?4
-?*
(?%
inputs?????????55$
? "-?*
#? 
0?????????0
? ?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_117459?_`I?F
??<
:?7
inputs+???????????????????????????$
? "??<
5?2
0+???????????????????????????0
? ?
)__inference_conv2d_2_layer_call_fn_117430__`7?4
-?*
(?%
inputs?????????55$
? " ??????????0?
)__inference_conv2d_2_layer_call_fn_117439?_`I?F
??<
:?7
inputs+???????????????????????????$
? "2?/+???????????????????????????0?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_117619n??7?4
-?*
(?%
inputs?????????0
? "-?*
#? 
0?????????

@
? ?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_117629???I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+???????????????????????????@
? ?
)__inference_conv2d_3_layer_call_fn_117600a??7?4
-?*
(?%
inputs?????????0
? " ??????????

@?
)__inference_conv2d_3_layer_call_fn_117609???I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+???????????????????????????@?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_117677n??7?4
-?*
(?%
inputs?????????

@
? "-?*
#? 
0?????????@
? ?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_117687???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
)__inference_conv2d_4_layer_call_fn_117658a??7?4
-?*
(?%
inputs?????????

@
? " ??????????@?
)__inference_conv2d_4_layer_call_fn_117667???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
B__inference_conv2d_layer_call_and_return_conditional_losses_117333n?@9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????nn
? ?
B__inference_conv2d_layer_call_and_return_conditional_losses_117343??@I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
'__inference_conv2d_layer_call_fn_117314a?@9?6
/?,
*?'
inputs???????????
? " ??????????nn?
'__inference_conv2d_layer_call_fn_117323??@I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
C__inference_dense_1_layer_call_and_return_conditional_losses_117839_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????@
? ~
(__inference_dense_1_layer_call_fn_117829R??0?-
&?#
!?
inputs??????????
? "??????????@?
C__inference_dense_2_layer_call_and_return_conditional_losses_117868^??/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? }
(__inference_dense_2_layer_call_fn_117858Q??/?,
%?"
 ?
inputs?????????@
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_117773`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
A__inference_dense_layer_call_and_return_conditional_losses_117783h??8?5
.?+
)?&
inputs??????????????????
? "&?#
?
0??????????
? }
&__inference_dense_layer_call_fn_117754S??0?-
&?#
!?
inputs??????????
? "????????????
&__inference_dense_layer_call_fn_117763[??8?5
.?+
)?&
inputs??????????????????
? "????????????
C__inference_dropout_layer_call_and_return_conditional_losses_117808^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_117820^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? }
(__inference_dropout_layer_call_fn_117798Q4?1
*?'
!?
inputs??????????
p 
? "???????????}
(__inference_dropout_layer_call_fn_117803Q4?1
*?'
!?
inputs??????????
p
? "????????????
C__inference_flatten_layer_call_and_return_conditional_losses_117733a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
C__inference_flatten_layer_call_and_return_conditional_losses_117745{I?F
??<
:?7
inputs+???????????????????????????@
? ".?+
$?!
0??????????????????
? ?
(__inference_flatten_layer_call_fn_117722T7?4
-?*
(?%
inputs?????????@
? "????????????
(__inference_flatten_layer_call_fn_117727nI?F
??<
:?7
inputs+???????????????????????????@
? "!????????????????????
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_117717?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_1_layer_call_fn_117712?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_117489?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_layer_call_fn_117484?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_model_layer_call_and_return_conditional_losses_116577?"4567?@OP_`wxyz??????????B??
8?5
+?(
input_1???????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_116654?$?4567?@OP_`wxyz??????????B??
8?5
+?(
input_1???????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_116872?"4567?@OP_`wxyz??????????A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_117080?$?4567?@OP_`wxyz??????????A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
&__inference_model_layer_call_fn_115870?"4567?@OP_`wxyz??????????B??
8?5
+?(
input_1???????????
p 

 
? "???????????
&__inference_model_layer_call_fn_116502?$?4567?@OP_`wxyz??????????B??
8?5
+?(
input_1???????????
p

 
? "???????????
&__inference_model_layer_call_fn_116713?"4567?@OP_`wxyz??????????A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
&__inference_model_layer_call_fn_116768?$?4567?@OP_`wxyz??????????A?>
7?4
*?'
inputs???????????
p

 
? "???????????
K__inference_random_contrast_layer_call_and_return_conditional_losses_117162p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
K__inference_random_contrast_layer_call_and_return_conditional_losses_117203t?=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
0__inference_random_contrast_layer_call_fn_117151c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
0__inference_random_contrast_layer_call_fn_117158g?=?:
3?0
*?'
inputs???????????
p
? ""?????????????
D__inference_resizing_layer_call_and_return_conditional_losses_117146l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_resizing_layer_call_fn_117140_9?6
/?,
*?'
inputs???????????
? ""?????????????
$__inference_signature_wrapper_117135?"4567?@OP_`wxyz??????????E?B
? 
;?8
6
input_1+?(
input_1???????????"1?.
,
dense_2!?
dense_2?????????