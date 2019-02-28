from building import *

cwd = GetCurrentDir()

Import('asenv')
MODULES = asenv['MODULES']

objs = []

objs += Glob('CMSIS/NN/Source/*/*.c')
objs += Glob('CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/*.cpp')
objs += Glob('CMSIS/NN/Examples/ARM/arm_nn_examples/mnist/code/*.c')

asenv.Append(CPPPATH=['%s/CMSIS/NN/Include'%(cwd),
                      '%s/CMSIS/DSP/Include'%(cwd),
                      '%s/CMSIS/Core/Include'%(cwd),])
asenv.Append(CPPDEFINES=['__ARM_ARCH_8M_BASE__'])

Return('objs')
