from building import *

cwd = GetCurrentDir()

Import('asenv')
MODULES = asenv['MODULES']

objs = []

env = ForkEnv(asenv)

objs += Glob('CMSIS/NN/Source/*/*.c')
objs += Glob('CMSIS/NN/Examples/ARM/arm_nn_examples/cifar10/*.cpp')

env.Append(CPPPATH=['%s/CMSIS/NN/Include'%(cwd),
                      '%s/CMSIS/DSP/Include'%(cwd),
                      '%s/CMSIS/Core/Include'%(cwd),])
env.Append(CPPDEFINES=['__ARM_ARCH_8M_BASE__'])

objs = env.Object(objs)

Return('objs')
