/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { ENGINE } from './engine';
import * as tf from './index';
import { ALL_ENVS, describeWithFlags, TestKernelBackend } from './jasmine_util';
import { expectArraysClose } from './test_util';
describe('Backend registration', () => {
    beforeAll(() => {
        // Silences backend registration warnings.
        spyOn(console, 'warn');
    });
    let registeredBackends = [];
    let registerBackend;
    beforeEach(() => {
        // Registering a backend changes global state (engine), so we wrap
        // registration to automatically remove registered backend at the end
        // of each test.
        registerBackend = (name, factory, priority) => {
            registeredBackends.push(name);
            return tf.registerBackend(name, factory, priority);
        };
        ENGINE.reset();
    });
    afterEach(() => {
        // Remove all registered backends at the end of each test.
        registeredBackends.forEach(name => {
            if (tf.findBackendFactory(name) != null) {
                tf.removeBackend(name);
            }
        });
        registeredBackends = [];
    });
    it('removeBackend disposes the backend and removes the factory', () => {
        let backend;
        const factory = () => {
            const newBackend = new TestKernelBackend();
            if (backend == null) {
                backend = newBackend;
                spyOn(backend, 'dispose').and.callThrough();
            }
            return newBackend;
        };
        registerBackend('test-backend', factory);
        expect(tf.findBackend('test-backend') != null).toBe(true);
        expect(tf.findBackend('test-backend')).toBe(backend);
        expect(tf.findBackendFactory('test-backend')).toBe(factory);
        tf.removeBackend('test-backend');
        expect(tf.findBackend('test-backend') == null).toBe(true);
        expect(tf.findBackend('test-backend')).toBe(null);
        expect(backend.dispose.calls.count()).toBe(1);
        expect(tf.findBackendFactory('test-backend')).toBe(null);
    });
    it('findBackend initializes the backend', () => {
        let backend;
        const factory = () => {
            const newBackend = new TestKernelBackend();
            if (backend == null) {
                backend = newBackend;
            }
            return newBackend;
        };
        registerBackend('custom-cpu', factory);
        expect(tf.findBackend('custom-cpu') != null).toBe(true);
        expect(tf.findBackend('custom-cpu')).toBe(backend);
        expect(tf.findBackendFactory('custom-cpu')).toBe(factory);
    });
    it('custom backend registration', () => {
        let backend;
        const priority = 103;
        registerBackend('custom-cpu', () => {
            const newBackend = new TestKernelBackend();
            if (backend == null) {
                backend = newBackend;
            }
            return newBackend;
        }, priority);
        expect(tf.backend() != null).toBe(true);
        expect(tf.backend()).toBe(backend);
    });
    it('high priority backend registration fails, falls back', () => {
        let lowPriorityBackend;
        const lowPriority = 103;
        const highPriority = 104;
        registerBackend('custom-low-priority', () => {
            lowPriorityBackend = new TestKernelBackend();
            return lowPriorityBackend;
        }, lowPriority);
        registerBackend('custom-high-priority', () => {
            throw new Error(`High priority backend fails`);
        }, highPriority);
        expect(tf.backend() != null).toBe(true);
        expect(tf.backend()).toBe(lowPriorityBackend);
        expect(tf.getBackend()).toBe('custom-low-priority');
    });
    it('low priority and high priority backends, setBackend low priority', () => {
        let lowPriorityBackend;
        let highPriorityBackend;
        const lowPriority = 103;
        const highPriority = 104;
        registerBackend('custom-low-priority', () => {
            lowPriorityBackend = new TestKernelBackend();
            return lowPriorityBackend;
        }, lowPriority);
        registerBackend('custom-high-priority', () => {
            highPriorityBackend = new TestKernelBackend();
            return highPriorityBackend;
        }, highPriority);
        expect(tf.backend() != null).toBe(true);
        expect(tf.backend()).toBe(highPriorityBackend);
        expect(tf.getBackend()).toBe('custom-high-priority');
        tf.setBackend('custom-low-priority');
        expect(tf.backend() != null).toBe(true);
        expect(tf.backend()).toBe(lowPriorityBackend);
        expect(tf.getBackend()).toBe('custom-low-priority');
    });
    it('default custom background null', () => {
        expect(tf.findBackend('custom')).toBeNull();
    });
    it('allow custom backend', () => {
        const backend = new TestKernelBackend();
        const success = registerBackend('custom', () => backend);
        expect(success).toBeTruthy();
        expect(tf.findBackend('custom')).toEqual(backend);
    });
    it('sync backend with await ready works', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('sync', () => testBackend);
        tf.setBackend('sync');
        expect(tf.getBackend()).toEqual('sync');
        await tf.ready();
        expect(tf.backend()).toEqual(testBackend);
    });
    it('sync backend without await ready works', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('sync', () => testBackend);
        tf.setBackend('sync');
        expect(tf.getBackend()).toEqual('sync');
        expect(tf.backend()).toEqual(testBackend);
    });
    it('async backend with await ready works', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async', async () => {
            await tf.nextFrame();
            return testBackend;
        });
        tf.setBackend('async');
        expect(tf.getBackend()).toEqual('async');
        await tf.ready();
        expect(tf.backend()).toEqual(testBackend);
    });
    it('async backend without await ready does not work', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async', async () => {
            await tf.nextFrame();
            return testBackend;
        });
        tf.setBackend('async');
        expect(tf.getBackend()).toEqual('async');
        expect(() => tf.backend())
            .toThrowError(/Backend 'async' has not yet been initialized./);
    });
    it('tf.square() fails if user does not await ready on async backend', async () => {
        registerBackend('async', async () => {
            await tf.nextFrame();
            return new TestKernelBackend();
        });
        tf.setBackend('async');
        expect(() => tf.square(2))
            .toThrowError(/Backend 'async' has not yet been initialized/);
    });
    it('tf.square() works when user awaits ready on async backend', async () => {
        registerBackend('async', async () => {
            await tf.nextFrame();
            return new TestKernelBackend();
        });
        tf.setBackend('async');
        await tf.ready();
        expect(() => tf.square(2)).toThrowError(/'write' not yet implemented/);
    });
    it('Registering async2 (higher priority) fails, async1 becomes active', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async1', async () => {
            await tf.nextFrame();
            return testBackend;
        }, 100 /* priority */);
        registerBackend('async2', async () => {
            await tf.nextFrame();
            throw new Error('failed to create async2');
        }, 101 /* priority */);
        // Await for the library to find the best backend that succesfully
        // initializes.
        await tf.ready();
        expect(tf.backend()).toEqual(testBackend);
        expect(tf.getBackend()).toBe('async1');
    });
    it('Registering sync as higher priority and async as lower priority', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('sync', () => testBackend, 101 /* priority */);
        registerBackend('async', async () => {
            await tf.nextFrame();
            return new TestKernelBackend();
        }, 100 /* priority */);
        // No need to await for ready() since the highest priority one is sync.
        expect(tf.backend()).toEqual(testBackend);
        expect(tf.getBackend()).toBe('sync');
    });
    it('async as higher priority and sync as lower priority with await ready', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async', async () => {
            await tf.nextFrame();
            return testBackend;
        }, 101 /* priority */);
        registerBackend('sync', () => new TestKernelBackend(), 100 /* priority */);
        await tf.ready();
        expect(tf.backend()).toEqual(testBackend);
        expect(tf.getBackend()).toBe('async');
    });
    it('async as higher priority and sync as lower priority w/o await ready', async () => {
        const testBackend = new TestKernelBackend();
        registerBackend('async', async () => {
            await tf.nextFrame();
            return testBackend;
        }, 101 /* priority */);
        registerBackend('sync', () => new TestKernelBackend(), 100 /* priority */);
        expect(() => tf.backend())
            .toThrowError(/The highest priority backend 'async' has not yet been/);
    });
    it('Registering and setting a backend that fails to register', async () => {
        registerBackend('async', async () => {
            await tf.nextFrame();
            throw new Error('failed to create async');
        });
        const success = tf.setBackend('async');
        expect(tf.getBackend()).toBe('async');
        expect(() => tf.backend())
            .toThrowError(/Backend 'async' has not yet been initialized/);
        expect(await success).toBe(false);
    });
});
describeWithFlags('memory', ALL_ENVS, () => {
    it('Sum(float)', async () => {
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numBytes).toBe(0);
        const sum = tf.tidy(() => {
            const a = tf.tensor1d([1, 2, 3, 4]);
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4 * 4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        expectArraysClose(await sum.data(), [1 + 2 + 3 + 4]);
    });
    it('Sum(bool)', async () => {
        const sum = tf.tidy(() => {
            const a = tf.tensor1d([true, true, false, true], 'bool');
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        expect(sum.dtype).toBe('int32');
        expectArraysClose(await sum.data(), [1 + 1 + 0 + 1]);
    });
    it('Sum(int32)', async () => {
        const sum = tf.tidy(() => {
            const a = tf.tensor1d([1, 1, 0, 1], 'int32');
            expect(tf.memory().numTensors).toBe(1);
            expect(tf.memory().numBytes).toBe(4 * 4);
            return a.sum();
        });
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(4);
        expect(sum.dtype).toBe('int32');
        expectArraysClose(await sum.data(), [1 + 1 + 0 + 1]);
    });
    it('string tensor', () => {
        const a = tf.tensor([['a', 'bb'], ['c', 'd']]);
        expect(tf.memory().numTensors).toBe(1);
        expect(tf.memory().numBytes).toBe(5); // 5 letters, each 1 byte in utf8.
        a.dispose();
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numBytes).toBe(0);
    });
    it('unreliable is true for string tensors', () => {
        tf.tensor('a');
        const mem = tf.memory();
        expect(mem.unreliable).toBe(true);
        const expectedReason = 'Memory usage by string tensors is approximate ' +
            '(2 bytes per character)';
        expect(mem.reasons.indexOf(expectedReason) >= 0).toBe(true);
    });
    it('makeTensorFromDataId creates a tensor', () => {
        const tensor = ENGINE.makeTensorFromDataId({}, [3], 'float32');
        expect(tensor).toBeDefined();
        expect(tensor.shape).toEqual([3]);
    });
});
describeWithFlags('profile', ALL_ENVS, () => {
    it('squaring', async () => {
        const profile = await tf.profile(() => {
            const x = tf.tensor1d([1, 2, 3]);
            let x2 = x.square();
            x2.dispose();
            x2 = x.square();
            x2.dispose();
            return x;
        });
        const result = profile.result;
        expect(profile.newBytes).toBe(12);
        expect(profile.peakBytes).toBe(24);
        expect(profile.newTensors).toBe(1);
        expectArraysClose(await result.data(), [1, 2, 3]);
        expect(profile.kernels.length).toBe(2);
        // Test the types for `kernelTimeMs` and `extraInfo` to confirm the promises
        // are resolved.
        expect(profile.kernels[0].kernelTimeMs instanceof Promise).toBe(false);
        expect(profile.kernels[0].extraInfo instanceof Promise).toBe(false);
        expect(profile.kernels[1].kernelTimeMs instanceof Promise).toBe(false);
        expect(profile.kernels[1].extraInfo instanceof Promise).toBe(false);
        // The specific values of `kernelTimeMs` and `extraInfo` are tested in the
        // tests of Profiler.profileKernel, so their values are not tested here.
        expect(profile.kernels[0]).toEqual({
            'name': 'Square',
            'bytesAdded': 12,
            'totalBytesSnapshot': 24,
            'tensorsAdded': 1,
            'totalTensorsSnapshot': 2,
            'inputShapes': [[3]],
            'outputShapes': [[3]],
            'kernelTimeMs': profile.kernels[0].kernelTimeMs,
            'extraInfo': profile.kernels[0].extraInfo
        });
        expect(profile.kernels[1]).toEqual({
            'name': 'Square',
            'bytesAdded': 12,
            'totalBytesSnapshot': 24,
            'tensorsAdded': 1,
            'totalTensorsSnapshot': 2,
            'inputShapes': [[3]],
            'outputShapes': [[3]],
            'kernelTimeMs': profile.kernels[1].kernelTimeMs,
            'extraInfo': profile.kernels[1].extraInfo
        });
    });
    it('squaring without disposing', async () => {
        const profile = await tf.profile(() => {
            const x = tf.tensor1d([1, 2, 3]);
            const x2 = x.square();
            return x2;
        });
        const result = profile.result;
        expect(profile.newBytes).toBe(24);
        expect(profile.peakBytes).toBe(24);
        expect(profile.newTensors).toBe(2);
        expectArraysClose(await result.data(), [1, 4, 9]);
        expect(profile.kernels.length).toBe(1);
        expect(profile.kernels[0].kernelTimeMs instanceof Promise).toBe(false);
        expect(profile.kernels[0].extraInfo instanceof Promise).toBe(false);
        expect(profile.kernels[0]).toEqual({
            'name': 'Square',
            'bytesAdded': 12,
            'totalBytesSnapshot': 24,
            'tensorsAdded': 1,
            'totalTensorsSnapshot': 2,
            'inputShapes': [[3]],
            'outputShapes': [[3]],
            'kernelTimeMs': profile.kernels[0].kernelTimeMs,
            'extraInfo': profile.kernels[0].extraInfo
        });
    });
    it('squaring in async query', async () => {
        const profile = await tf.profile(async () => {
            await new Promise(resolve => setTimeout(resolve, 1));
            const x = tf.tensor1d([1, 2, 3]);
            const x2 = x.square();
            x2.dispose();
            return x;
        });
        const result = profile.result;
        expect(profile.newBytes).toBe(12);
        expect(profile.peakBytes).toBe(24);
        expect(profile.newTensors).toBe(1);
        expectArraysClose(await result.data(), [1, 2, 3]);
        expect(profile.kernels.length).toBe(1);
        expect(profile.kernels[0].kernelTimeMs instanceof Promise).toBe(false);
        expect(profile.kernels[0].extraInfo instanceof Promise).toBe(false);
        expect(profile.kernels[0]).toEqual({
            'name': 'Square',
            'bytesAdded': 12,
            'totalBytesSnapshot': 24,
            'tensorsAdded': 1,
            'totalTensorsSnapshot': 2,
            'inputShapes': [[3]],
            'outputShapes': [[3]],
            'kernelTimeMs': profile.kernels[0].kernelTimeMs,
            'extraInfo': profile.kernels[0].extraInfo
        });
    });
    it('reports correct kernelNames', async () => {
        const profile = await tf.profile(() => {
            const x = tf.tensor1d([1, 2, 3]);
            const x2 = x.square();
            const x3 = x2.abs();
            return x3;
        });
        expect(profile.kernelNames).toEqual(jasmine.arrayWithExactContents([
            'Square', 'Abs'
        ]));
    });
});
describeWithFlags('disposeVariables', ALL_ENVS, () => {
    it('reuse same name variable', () => {
        tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        tf.tensor1d([1, 2, 3]).variable(true, 'v2');
        expect(() => {
            tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        }).toThrowError();
        tf.disposeVariables();
        tf.tensor1d([1, 2, 3]).variable(true, 'v1');
        tf.tensor1d([1, 2, 3]).variable(true, 'v2');
    });
});
/**
 * The following test constraints to the CPU environment because it needs a
 * concrete backend to exist. This test will work for any backend, but currently
 * this is the simplest backend to test against.
 */
describeWithFlags('Switching cpu backends', { predicate: testEnv => testEnv.backendName === 'cpu' }, () => {
    beforeEach(() => {
        tf.registerBackend('cpu1', tf.findBackendFactory('cpu'));
        tf.registerBackend('cpu2', tf.findBackendFactory('cpu'));
    });
    afterEach(() => {
        tf.removeBackend('cpu1');
        tf.removeBackend('cpu2');
    });
    it('Move data from cpu1 to cpu2 backend', async () => {
        tf.setBackend('cpu1');
        // This scalar lives in cpu1.
        const a = tf.scalar(5);
        tf.setBackend('cpu2');
        // This scalar lives in cpu2.
        const b = tf.scalar(3);
        expect(tf.memory().numDataBuffers).toBe(2);
        expect(tf.memory().numTensors).toBe(2);
        expect(tf.memory().numBytes).toBe(8);
        // Make sure you can read both tensors.
        expectArraysClose(await a.data(), [5]);
        expectArraysClose(await b.data(), [3]);
        // Switch back to cpu1.
        tf.setBackend('cpu1');
        // Again make sure you can read both tensors.
        expectArraysClose(await a.data(), [5]);
        expectArraysClose(await b.data(), [3]);
        tf.dispose([a, b]);
        expect(tf.memory().numDataBuffers).toBe(0);
        expect(tf.memory().numTensors).toBe(0);
        expect(tf.memory().numBytes).toBe(0);
    });
    it('can execute op with data from mixed backends', async () => {
        const kernelFunc = tf.getKernel('Add', 'cpu').kernelFunc;
        tf.registerKernel({ kernelName: 'Add', backendName: 'cpu1', kernelFunc });
        tf.registerKernel({ kernelName: 'Add', backendName: 'cpu2', kernelFunc });
        tf.setBackend('cpu1');
        // This scalar lives in cpu1.
        const a = tf.scalar(5);
        tf.setBackend('cpu2');
        // This scalar lives in cpu2.
        const b = tf.scalar(3);
        // Verify that ops can execute with mixed backend data.
        ENGINE.startScope();
        tf.setBackend('cpu1');
        expectArraysClose(await tf.add(a, b).data(), [8]);
        tf.setBackend('cpu2');
        expectArraysClose(await tf.add(a, b).data(), [8]);
        ENGINE.endScope();
        tf.dispose([a, b]);
    });
});
describeWithFlags('Detects memory leaks in kernels', ALL_ENVS, () => {
    const backendName = 'test-mem';
    const kernelName = 'MyKernel';
    const kernelNameComplex = 'Kernel-complex';
    it('Detects memory leak in a kernel', () => {
        let dataIdsCount = 0;
        tf.registerBackend(backendName, () => {
            return {
                id: 1,
                dispose: () => null,
                disposeData: (dataId) => null,
                numDataIds: () => dataIdsCount
            };
        });
        const kernelWithMemLeak = () => {
            dataIdsCount += 2;
            return { dataId: {}, shape: [], dtype: 'float32' };
        };
        tf.registerKernel({ kernelName, backendName, kernelFunc: kernelWithMemLeak });
        tf.setBackend(backendName);
        expect(() => tf.engine().runKernel(kernelName, {}, {}))
            .toThrowError(/Backend 'test-mem' has an internal memory leak \(1 data ids\)/);
        tf.removeBackend(backendName);
        tf.unregisterKernel(kernelName, backendName);
    });
    it('No mem leak in a kernel with multiple outputs', () => {
        let dataIdsCount = 0;
        tf.registerBackend(backendName, () => {
            return {
                id: 1,
                dispose: () => null,
                disposeData: (dataId) => null,
                numDataIds: () => dataIdsCount
            };
        });
        tf.setBackend(backendName);
        const kernelWith3Outputs = () => {
            dataIdsCount += 3;
            const t = { dataId: {}, shape: [], dtype: 'float32' };
            return [t, t, t];
        };
        tf.registerKernel({ kernelName, backendName, kernelFunc: kernelWith3Outputs });
        const res = tf.engine().runKernel(kernelName, {}, {});
        expect(Array.isArray(res)).toBe(true);
        expect(res.length).toBe(3);
        const kernelWithComplexOutputs = () => {
            dataIdsCount += 3;
            return { dataId: {}, shape: [], dtype: 'complex64' };
        };
        tf.registerKernel({
            kernelName: kernelNameComplex,
            backendName,
            kernelFunc: kernelWithComplexOutputs
        });
        const res2 = tf.engine().runKernel(kernelNameComplex, {}, {});
        expect(res2.shape).toEqual([]);
        expect(res2.dtype).toEqual('complex64');
        tf.removeBackend(backendName);
        tf.unregisterKernel(kernelName, backendName);
        tf.unregisterKernel(kernelNameComplex, backendName);
    });
});
// NOTE: This describe is purposefully not a describeWithFlags so that we
// test tensor allocation where no scopes have been created.
describe('Memory allocation outside a test scope', () => {
    it('constructing a tensor works', async () => {
        const backendName = 'test-backend';
        tf.registerBackend(backendName, () => {
            let storedValues = null;
            return {
                id: 1,
                floatPrecision: () => 32,
                write: (values, shape, dtype) => {
                    const dataId = {};
                    storedValues = values;
                    return dataId;
                },
                read: async (dataId) => storedValues,
                dispose: () => null,
                disposeData: (dataId) => null
            };
        });
        tf.setBackend(backendName);
        const a = tf.tensor1d([1, 2, 3]);
        expectArraysClose(await a.data(), [1, 2, 3]);
        a.dispose();
        tf.removeBackend(backendName);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZW5naW5lX3Rlc3QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL2VuZ2luZV90ZXN0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDaEMsT0FBTyxLQUFLLEVBQUUsTUFBTSxTQUFTLENBQUM7QUFFOUIsT0FBTyxFQUFDLFFBQVEsRUFBRSxpQkFBaUIsRUFBRSxpQkFBaUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBRzlFLE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUc5QyxRQUFRLENBQUMsc0JBQXNCLEVBQUUsR0FBRyxFQUFFO0lBQ3BDLFNBQVMsQ0FBQyxHQUFHLEVBQUU7UUFDYiwwQ0FBMEM7UUFDMUMsS0FBSyxDQUFDLE9BQU8sRUFBRSxNQUFNLENBQUMsQ0FBQztJQUN6QixDQUFDLENBQUMsQ0FBQztJQUVILElBQUksa0JBQWtCLEdBQWEsRUFBRSxDQUFDO0lBQ3RDLElBQUksZUFBMEMsQ0FBQztJQUUvQyxVQUFVLENBQUMsR0FBRyxFQUFFO1FBQ2Qsa0VBQWtFO1FBQ2xFLHFFQUFxRTtRQUNyRSxnQkFBZ0I7UUFDaEIsZUFBZSxHQUFHLENBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxRQUFRLEVBQUUsRUFBRTtZQUM1QyxrQkFBa0IsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDOUIsT0FBTyxFQUFFLENBQUMsZUFBZSxDQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsUUFBUSxDQUFDLENBQUM7UUFDckQsQ0FBQyxDQUFDO1FBRUYsTUFBTSxDQUFDLEtBQUssRUFBRSxDQUFDO0lBQ2pCLENBQUMsQ0FBQyxDQUFDO0lBRUgsU0FBUyxDQUFDLEdBQUcsRUFBRTtRQUNiLDBEQUEwRDtRQUMxRCxrQkFBa0IsQ0FBQyxPQUFPLENBQUMsSUFB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QUFDLFFBQVEsRUFBRSxRQUFRLEVBQUUsR0FBRyxFQUFFO0lBQ3pDLEVBQUUsQ0FBQyxZQUFZLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDMUIsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDckMsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDdkIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDcEMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdkMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO1lBQ3pDLE9BQU8sQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ2pCLENBQUMsQ0FBQyxDQUFDO1FBQ0gsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDckMsaUJBQWlCLENBQUMsTUFBTSxHQUFHLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLFdBQVcsRUFBRSxLQUFLLElBQUksRUFBRTtRQUN6QixNQUFNLEdBQUcsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRTtZQUN2QixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxFQUFFLElBQUksRUFBRSxLQUFLLEVBQUUsSUFBSSxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUM7WUFDekQsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdkMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDckMsT0FBTyxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDakIsQ0FBQyxDQUFDLENBQUM7UUFDSCxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN2QyxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNyQyxNQUFNLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNoQyxpQkFBaUIsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsWUFBWSxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzFCLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO1lBQ3ZCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztZQUM3QyxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN2QyxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7WUFDekMsT0FBTyxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDakIsQ0FBQyxDQUFDLENBQUM7UUFDSCxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN2QyxNQUFNLENBQUMsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNyQyxNQUFNLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUNoQyxpQkFBaUIsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsZUFBZSxFQUFFLEdBQUcsRUFBRTtRQUN2QixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLEVBQUUsSUFBSSxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRS9DLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3ZDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUUsa0NBQWtDO1FBRXpFLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUVaLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3ZDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHVDQUF1QyxFQUFFLEdBQUcsRUFBRTtRQUMvQyxFQUFFLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ2YsTUFBTSxHQUFHLEdBQUcsRUFBRSxDQUFDLE1BQU0sRUFBRSxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxHQUFHLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ2xDLE1BQU0sY0FBYyxHQUFHLGdEQUFnRDtZQUNuRSx5QkFBeUIsQ0FBQztRQUM5QixNQUFNLENBQUMsR0FBRyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQzlELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHVDQUF1QyxFQUFFLEdBQUcsRUFBRTtRQUMvQyxNQUFNLE1BQU0sR0FBRyxNQUFNLENBQUMsb0JBQW9CLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUM7UUFDL0QsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLFdBQVcsRUFBRSxDQUFDO1FBQzdCLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNwQyxDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDO0FBRUgsaUJBQWlCLENBQUMsU0FBUyxFQUFFLFFBQVEsRUFBRSxHQUFHLEVBQUU7SUFDMUMsRUFBRSxDQUFDLFVBQVUsRUFBRSxLQUFLLElBQUksRUFBRTtRQUN4QixNQUFNLE9BQU8sR0FBRyxNQUFNLEVBQUUsQ0FBQyxPQUFPLENBQUMsR0FBRyxFQUFFO1lBQ3BDLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDakMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxDQUFDLE1BQU0sRUFBRSxDQUFDO1lBQ3BCLEVBQUUsQ0FBQyxPQUFPLEVBQUUsQ0FBQztZQUNiLEVBQUUsR0FBRyxDQUFDLENBQUMsTUFBTSxFQUFFLENBQUM7WUFDaEIsRUFBRSxDQUFDLE9BQU8sRUFBRSxDQUFDO1lBQ2IsT0FBTyxDQUFDLENBQUM7UUFDWCxDQUFDLENBQUMsQ0FBQztRQUVILE1BQU0sTUFBTSxHQUFHLE9BQU8sQ0FBQyxNQUFnQixDQUFDO1FBRXhDLE1BQU0sQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBQ2xDLE1BQU0sQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBQ25DLE1BQU0sQ0FBQyxPQUFPLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ25DLGlCQUFpQixDQUFDLE1BQU0sTUFBTSxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2xELE1BQU0sQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUV2Qyw0RUFBNEU7UUFDNUUsZ0JBQWdCO1FBQ2hCLE1BQU0sQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLFlBQVksWUFBWSxPQUFPLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDdkUsTUFBTSxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsU0FBUyxZQUFZLE9BQU8sQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUNwRSxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxZQUFZLFlBQVksT0FBTyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3ZFLE1BQU0sQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLFNBQVMsWUFBWSxPQUFPLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFFcEUsMEVBQTBFO1FBQzFFLHdFQUF3RTtRQUN4RSxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQztZQUNqQyxNQUFNLEVBQUUsUUFBUTtZQUNoQixZQUFZLEVBQUUsRUFBRTtZQUNoQixvQkFBb0IsRUFBRSxFQUFFO1lBQ3hCLGNBQWMsRUFBRSxDQUFDO1lBQ2pCLHNCQUFzQixFQUFFLENBQUM7WUFDekIsYUFBYSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNwQixjQUFjLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3JCLGNBQWMsRUFBRSxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLFlBQVk7WUFDL0MsV0FBVyxFQUFFLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsU0FBUztTQUMxQyxDQUFDLENBQUM7UUFFSCxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQztZQUNqQyxNQUFNLEVBQUUsUUFBUTtZQUNoQixZQUFZLEVBQUUsRUFBRTtZQUNoQixvQkFBb0IsRUFBRSxFQUFFO1lBQ3hCLGNBQWMsRUFBRSxDQUFDO1lBQ2pCLHNCQUFzQixFQUFFLENBQUM7WUFDekIsYUFBYSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNwQixjQUFjLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3JCLGNBQWMsRUFBRSxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLFlBQVk7WUFDL0MsV0FBVyxFQUFFLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsU0FBUztTQUMxQyxDQUFDLENBQUM7SUFDTCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyw0QkFBNEIsRUFBRSxLQUFLLElBQUksRUFBRTtRQUMxQyxNQUFNLE9BQU8sR0FBRyxNQUFNLEVBQUUsQ0FBQyxPQUFPLENBQUMsR0FBRyxFQUFFO1lBQ3BDLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDakMsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDLE1BQU0sRUFBRSxDQUFDO1lBQ3RCLE9BQU8sRUFBRSxDQUFDO1FBQ1osQ0FBQyxDQUFDLENBQUM7UUFFSCxNQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsTUFBZ0IsQ0FBQztRQUV4QyxNQUFNLENBQUMsT0FBTyxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUNsQyxNQUFNLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUNuQyxNQUFNLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNuQyxpQkFBaUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNsRCxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsWUFBWSxZQUFZLE9BQU8sQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUN2RSxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTLFlBQVksT0FBTyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDO1lBQ2pDLE1BQU0sRUFBRSxRQUFRO1lBQ2hCLFlBQVksRUFBRSxFQUFFO1lBQ2hCLG9CQUFvQixFQUFFLEVBQUU7WUFDeEIsY0FBYyxFQUFFLENBQUM7WUFDakIsc0JBQXNCLEVBQUUsQ0FBQztZQUN6QixhQUFhLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3BCLGNBQWMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDckIsY0FBYyxFQUFFLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsWUFBWTtZQUMvQyxXQUFXLEVBQUUsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTO1NBQzFDLENBQUMsQ0FBQztJQUNMLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHlCQUF5QixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3ZDLE1BQU0sT0FBTyxHQUFHLE1BQU0sRUFBRSxDQUFDLE9BQU8sQ0FBQyxLQUFLLElBQUksRUFBRTtZQUMxQyxNQUFNLElBQUksT0FBTyxDQUFDLE9BQU8sQ0FBQyxFQUFFLENBQUMsVUFBVSxDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3JELE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDakMsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDLE1BQU0sRUFBRSxDQUFDO1lBQ3RCLEVBQUUsQ0FBQyxPQUFPLEVBQUUsQ0FBQztZQUNiLE9BQU8sQ0FBQyxDQUFDO1FBQ1gsQ0FBQyxDQUFDLENBQUM7UUFFSCxNQUFNLE1BQU0sR0FBRyxPQUFPLENBQUMsTUFBZ0IsQ0FBQztRQUV4QyxNQUFNLENBQUMsT0FBTyxDQUFDLFFBQVEsQ0FBQyxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUNsQyxNQUFNLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUNuQyxNQUFNLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNuQyxpQkFBaUIsQ0FBQyxNQUFNLE1BQU0sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNsRCxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsWUFBWSxZQUFZLE9BQU8sQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUN2RSxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTLFlBQVksT0FBTyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDO1lBQ2pDLE1BQU0sRUFBRSxRQUFRO1lBQ2hCLFlBQVksRUFBRSxFQUFFO1lBQ2hCLG9CQUFvQixFQUFFLEVBQUU7WUFDeEIsY0FBYyxFQUFFLENBQUM7WUFDakIsc0JBQXNCLEVBQUUsQ0FBQztZQUN6QixhQUFhLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3BCLGNBQWMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDckIsY0FBYyxFQUFFLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsWUFBWTtZQUMvQyxXQUFXLEVBQUUsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTO1NBQzFDLENBQUMsQ0FBQztJQUNMLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDZCQUE2QixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzNDLE1BQU0sT0FBTyxHQUFHLE1BQU0sRUFBRSxDQUFDLE9BQU8sQ0FBQyxHQUFHLEVBQUU7WUFDcEMsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNqQyxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsTUFBTSxFQUFFLENBQUM7WUFDdEIsTUFBTSxFQUFFLEdBQUcsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDO1lBQ3BCLE9BQU8sRUFBRSxDQUFDO1FBQ1osQ0FBQyxDQUFDLENBQUM7UUFFSCxNQUFNLENBQUMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsc0JBQXNCLENBQUM7WUFDakUsUUFBUSxFQUFFLEtBQUs7U0FDaEIsQ0FBQyxDQUFDLENBQUM7SUFDTixDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDO0FBRUgsaUJBQWlCLENBQUMsa0JBQWtCLEVBQUUsUUFBUSxFQUFFLEdBQUcsRUFBRTtJQUNuRCxFQUFFLENBQUMsMEJBQTBCLEVBQUUsR0FBRyxFQUFFO1FBQ2xDLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsUUFBUSxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztRQUM1QyxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDNUMsTUFBTSxDQUFDLEdBQUcsRUFBRTtZQUNWLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsUUFBUSxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztRQUM5QyxDQUFDLENBQUMsQ0FBQyxZQUFZLEVBQUUsQ0FBQztRQUNsQixFQUFFLENBQUMsZ0JBQWdCLEVBQUUsQ0FBQztRQUN0QixFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDNUMsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxRQUFRLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzlDLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUM7QUFFSDs7OztHQUlHO0FBQ0gsaUJBQWlCLENBQ2Isd0JBQXdCLEVBQ3hCLEVBQUMsU0FBUyxFQUFFLE9BQU8sQ0FBQyxFQUFFLENBQUMsT0FBTyxDQUFDLFdBQVcsS0FBSyxLQUFLLEVBQUMsRUFBRSxHQUFHLEVBQUU7SUFDMUQsVUFBVSxDQUFDLEdBQUcsRUFBRTtRQUNkLEVBQUUsQ0FBQyxlQUFlLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxrQkFBa0IsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO1FBQ3pELEVBQUUsQ0FBQyxlQUFlLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxrQkFBa0IsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO0lBQzNELENBQUMsQ0FBQyxDQUFDO0lBRUgsU0FBUyxDQUFDLEdBQUcsRUFBRTtRQUNiLEVBQUUsQ0FBQyxhQUFhLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDekIsRUFBRSxDQUFDLGFBQWEsQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUMzQixDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxxQ0FBcUMsRUFBRSxLQUFLLElBQUksRUFBRTtRQUNuRCxFQUFFLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3RCLDZCQUE2QjtRQUM3QixNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRXZCLEVBQUUsQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDdEIsNkJBQTZCO1FBQzdCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdkIsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxjQUFjLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDM0MsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFckMsdUNBQXVDO1FBQ3ZDLGlCQUFpQixDQUFDLE1BQU0sQ0FBQyxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN2QyxpQkFBaUIsQ0FBQyxNQUFNLENBQUMsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdkMsdUJBQXVCO1FBQ3ZCLEVBQUUsQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDdEIsNkNBQTZDO1FBQzdDLGlCQUFpQixDQUFDLE1BQU0sQ0FBQyxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN2QyxpQkFBaUIsQ0FBQyxNQUFNLENBQUMsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdkMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRW5CLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsY0FBYyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzNDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsVUFBVSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3ZDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDhDQUE4QyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQzVELE1BQU0sVUFBVSxHQUFHLEVBQUUsQ0FBQyxTQUFTLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDLFVBQVUsQ0FBQztRQUN6RCxFQUFFLENBQUMsY0FBYyxDQUFDLEVBQUMsVUFBVSxFQUFFLEtBQUssRUFBRSxXQUFXLEVBQUUsTUFBTSxFQUFFLFVBQVUsRUFBQyxDQUFDLENBQUM7UUFDeEUsRUFBRSxDQUFDLGNBQWMsQ0FBQyxFQUFDLFVBQVUsRUFBRSxLQUFLLEVBQUUsV0FBVyxFQUFFLE1BQU0sRUFBRSxVQUFVLEVBQUMsQ0FBQyxDQUFDO1FBRXhFLEVBQUUsQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDdEIsNkJBQTZCO1FBQzdCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdkIsRUFBRSxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN0Qiw2QkFBNkI7UUFDN0IsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUV2Qix1REFBdUQ7UUFDdkQsTUFBTSxDQUFDLFVBQVUsRUFBRSxDQUFDO1FBQ3BCLEVBQUUsQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDdEIsaUJBQWlCLENBQUMsTUFBTSxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFbEQsRUFBRSxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN0QixpQkFBaUIsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNsRCxNQUFNLENBQUMsUUFBUSxFQUFFLENBQUM7UUFFbEIsRUFBRSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3JCLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUM7QUEwR1AsaUJBQWlCLENBQUMsaUNBQWlDLEVBQUUsUUFBUSxFQUFFLEdBQUcsRUFBRTtJQUNsRSxNQUFNLFdBQVcsR0FBRyxVQUFVLENBQUM7SUFDL0IsTUFBTSxVQUFVLEdBQUcsVUFBVSxDQUFDO0lBQzlCLE1BQU0saUJBQWlCLEdBQUcsZ0JBQWdCLENBQUM7SUFFM0MsRUFBRSxDQUFDLGlDQUFpQyxFQUFFLEdBQUcsRUFBRTtRQUN6QyxJQUFJLFlBQVksR0FBRyxDQUFDLENBQUM7UUFDckIsRUFBRSxDQUFDLGVBQWUsQ0FBQyxXQUFXLEVBQUUsR0FBRyxFQUFFO1lBQ25DLE9BQU87Z0JBQ0wsRUFBRSxFQUFFLENBQUM7Z0JBQ0wsT0FBTyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUk7Z0JBQ25CLFdBQVcsRUFBRSxDQUFDLE1BQVUsRUFBRSxFQUFFLENBQUMsSUFBSTtnQkFDakMsVUFBVSxFQUFFLEdBQUcsRUFBRSxDQUFDLFlBQVk7YUFDaEIsQ0FBQztRQUNuQixDQUFDLENBQUMsQ0FBQztRQUVILE1BQU0saUJBQWlCLEdBQWUsR0FBRyxFQUFFO1lBQ3pDLFlBQVksSUFBSSxDQUFDLENBQUM7WUFDbEIsT0FBTyxFQUFDLE1BQU0sRUFBRSxFQUFFLEVBQUUsS0FBSyxFQUFFLEVBQUUsRUFBRSxLQUFLLEVBQUUsU0FBUyxFQUFDLENBQUM7UUFDbkQsQ0FBQyxDQUFDO1FBQ0YsRUFBRSxDQUFDLGNBQWMsQ0FBQyxFQUFDLFVBQVUsRUFBRSxXQUFXLEVBQUUsVUFBVSxFQUFFLGlCQUFpQixFQUFDLENBQUMsQ0FBQztRQUU1RSxFQUFFLENBQUMsVUFBVSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBQzNCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsU0FBUyxDQUFDLFVBQVUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7YUFDbEQsWUFBWSxDQUNULCtEQUErRCxDQUFDLENBQUM7UUFFekUsRUFBRSxDQUFDLGFBQWEsQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUM5QixFQUFFLENBQUMsZ0JBQWdCLENBQUMsVUFBVSxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQy9DLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLCtDQUErQyxFQUFFLEdBQUcsRUFBRTtRQUN2RCxJQUFJLFlBQVksR0FBRyxDQUFDLENBQUM7UUFDckIsRUFBRSxDQUFDLGVBQWUsQ0FBQyxXQUFXLEVBQUUsR0FBRyxFQUFFO1lBQ25DLE9BQU87Z0JBQ0wsRUFBRSxFQUFFLENBQUM7Z0JBQ0wsT0FBTyxFQUFFLEdBQUcsRUFBRSxDQUFDLElBQUk7Z0JBQ25CLFdBQVcsRUFBRSxDQUFDLE1BQVUsRUFBRSxFQUFFLENBQUMsSUFBSTtnQkFDakMsVUFBVSxFQUFFLEdBQUcsRUFBRSxDQUFDLFlBQVk7YUFDaEIsQ0FBQztRQUNuQixDQUFDLENBQUMsQ0FBQztRQUNILEVBQUUsQ0FBQyxVQUFVLENBQUMsV0FBVyxDQUFDLENBQUM7UUFFM0IsTUFBTSxrQkFBa0IsR0FBZSxHQUFHLEVBQUU7WUFDMUMsWUFBWSxJQUFJLENBQUMsQ0FBQztZQUNsQixNQUFNLENBQUMsR0FBZSxFQUFDLE1BQU0sRUFBRSxFQUFFLEVBQUUsS0FBSyxFQUFFLEVBQUUsRUFBRSxLQUFLLEVBQUUsU0FBUyxFQUFDLENBQUM7WUFDaEUsT0FBTyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDbkIsQ0FBQyxDQUFDO1FBQ0YsRUFBRSxDQUFDLGNBQWMsQ0FDYixFQUFDLFVBQVUsRUFBRSxXQUFXLEVBQUUsVUFBVSxFQUFFLGtCQUFrQixFQUFDLENBQUMsQ0FBQztRQUUvRCxNQUFNLEdBQUcsR0FBRyxFQUFFLENBQUMsTUFBTSxFQUFFLENBQUMsU0FBUyxDQUFDLFVBQVUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7UUFDdEQsTUFBTSxDQUFDLEtBQUssQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDdEMsTUFBTSxDQUFFLEdBQWlCLENBQUMsTUFBTSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTFDLE1BQU0sd0JBQXdCLEdBQWUsR0FBRyxFQUFFO1lBQ2hELFlBQVksSUFBSSxDQUFDLENBQUM7WUFDbEIsT0FBTyxFQUFDLE1BQU0sRUFBRSxFQUFFLEVBQUUsS0FBSyxFQUFFLEVBQUUsRUFBRSxLQUFLLEVBQUUsV0FBVyxFQUFDLENBQUM7UUFDckQsQ0FBQyxDQUFDO1FBQ0YsRUFBRSxDQUFDLGNBQWMsQ0FBQztZQUNoQixVQUFVLEVBQUUsaUJBQWlCO1lBQzdCLFdBQVc7WUFDWCxVQUFVLEVBQUUsd0JBQXdCO1NBQ3JDLENBQUMsQ0FBQztRQUVILE1BQU0sSUFBSSxHQUFHLEVBQUUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxTQUFTLENBQUMsaUJBQWlCLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBZSxDQUFDO1FBQzVFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBRXhDLEVBQUUsQ0FBQyxhQUFhLENBQUMsV0FBVyxDQUFDLENBQUM7UUFDOUIsRUFBRSxDQUFDLGdCQUFnQixDQUFDLFVBQVUsRUFBRSxXQUFXLENBQUMsQ0FBQztRQUM3QyxFQUFFLENBQUMsZ0JBQWdCLENBQUMsaUJBQWlCLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDdEQsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILHlFQUF5RTtBQUN6RSw0REFBNEQ7QUFDNUQsUUFBUSxDQUFDLHdDQUF3QyxFQUFFLEdBQUcsRUFBRTtJQUN0RCxFQUFFLENBQUMsNkJBQTZCLEVBQUUsS0FBSyxJQUFJLEVBQUU7UUFDM0MsTUFBTSxXQUFXLEdBQUcsY0FBYyxDQUFDO1FBQ25DLEVBQUUsQ0FBQyxlQUFlLENBQUMsV0FBVyxFQUFFLEdBQUcsRUFBRTtZQUNuQyxJQUFJLFlBQVksR0FBa0IsSUFBSSxDQUFDO1lBQ3ZDLE9BQU87Z0JBQ0wsRUFBRSxFQUFFLENBQUM7Z0JBQ0wsY0FBYyxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUU7Z0JBQ3hCLEtBQUssRUFBRSxDQUFDLE1BQXFCLEVBQUUsS0FBZSxFQUFFLEtBQWUsRUFBRSxFQUFFO29CQUNqRSxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUM7b0JBQ2xCLFlBQVksR0FBRyxNQUFNLENBQUM7b0JBQ3RCLE9BQU8sTUFBTSxDQUFDO2dCQUNoQixDQUFDO2dCQUNELElBQUksRUFBRSxLQUFLLEVBQUUsTUFBYyxFQUFFLEVBQUUsQ0FBQyxZQUFZO2dCQUM1QyxPQUFPLEVBQUUsR0FBRyxFQUFFLENBQUMsSUFBSTtnQkFDbkIsV0FBVyxFQUFFLENBQUMsTUFBVSxFQUFFLEVBQUUsQ0FBQyxJQUFJO2FBQ25CLENBQUM7UUFDbkIsQ0FBQyxDQUFDLENBQUM7UUFDSCxFQUFFLENBQUMsVUFBVSxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBRTNCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDakMsaUJBQWlCLENBQUMsTUFBTSxDQUFDLENBQUMsSUFBSSxFQUFFLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDN0MsQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO1FBRVosRUFBRSxDQUFDLGFBQWEsQ0FBQyxXQUFXLENBQUMsQ0FBQztJQUNoQyxDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbEJhY2tlbmR9IGZyb20gJy4vYmFja2VuZHMvYmFja2VuZCc7XG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi9lbmdpbmUnO1xuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi9pbmRleCc7XG5pbXBvcnQge0tlcm5lbEZ1bmN9IGZyb20gJy4vaW5kZXgnO1xuaW1wb3J0IHtBTExfRU5WUywgZGVzY3JpYmVXaXRoRmxhZ3MsIFRlc3RLZXJuZWxCYWNrZW5kfSBmcm9tICcuL2phc21pbmVfdXRpbCc7XG5pbXBvcnQgeyBUZW5zb3JJbmZvIH0gZnJvbSAnLi90ZW5zb3JfaW5mbyc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi90ZW5zb3InO1xuaW1wb3J0IHtleHBlY3RBcnJheXNDbG9zZX0gZnJvbSAnLi90ZXN0X3V0aWwnO1xuaW1wb3J0IHtCYWNrZW5kVmFsdWVzLCBEYXRhVHlwZX0gZnJvbSAnLi90eXBlcyc7XG5cbmRlc2NyaWJlKCdCYWNrZW5kIHJlZ2lzdHJhdGlvbicsICgpID0+IHtcbiAgYmVmb3JlQWxsKCgpID0+IHtcbiAgICAvLyBTaWxlbmNlcyBiYWNrZW5kIHJlZ2lzdHJhdGlvbiB3YXJuaW5ncy5cbiAgICBzcHlPbihjb25zb2xlLCAnd2FybicpO1xuICB9KTtcblxuICBsZXQgcmVnaXN0ZXJlZEJhY2tlbmRzOiBzdHJpbmdbXSA9IFtdO1xuICBsZXQgcmVnaXN0ZXJCYWNrZW5kOiB0eXBlb2YgdGYucmVnaXN0ZXJCYWNrZW5kO1xuXG4gIGJlZm9yZUVhY2goKCkgPT4ge1xuICAgIC8vIFJlZ2lzdGVyaW5nIGEgYmFja2VuZCBjaGFuZ2VzIGdsb2JhbCBzdGF0ZSAoZW5naW5lKSwgc28gd2Ugd3JhcFxuICAgIC8vIHJlZ2lzdHJhdGlvbiB0byBhdXRvbWF0aWNhbGx5IHJlbW92ZSByZWdpc3RlcmVkIGJhY2tlbmQgYXQgdGhlIGVuZFxuICAgIC8vIG9mIGVhY2ggdGVzdC5cbiAgICByZWdpc3RlckJhY2tlbmQgPSAobmFtZSwgZmFjdG9yeSwgcHJpb3JpdHkpID0+IHtcbiAgICAgIHJlZ2lzdGVyZWRCYWNrZW5kcy5wdXNoKG5hbWUpO1xuICAgICAgcmV0dXJuIHRmLnJlZ2lzdGVyQmFja2VuZChuYW1lLCBmYWN0b3J5LCBwcmlvcml0eSk7XG4gICAgfTtcblxuICAgIEVOR0lORS5yZXNldCgpO1xuICB9KTtcblxuICBhZnRlckVhY2goKCkgPT4ge1xuICAgIC8vIFJlbW92ZSBhbGwgcmVnaXN0ZXJlZCBiYWNrZW5kcyBhdCB0aGUgZW5kIG9mIGVhY2ggdGVzdC5cbiAgICByZWdpc3RlcmVkQmFja2VuZHMuZm9yRWFjaChuYW1lID0+IHtcbiAgICAgIGlmICh0Zi5maW5kQmFja2VuZEZhY3RvcnkobmFtZSkgIT0gbnVsbCkge1xuICAgICAgICB0Zi5yZW1vdmVCYWNrZW5kKG5hbWUpO1xuICAgICAgfVxuICAgIH0pO1xuICAgIHJlZ2lzdGVyZWRCYWNrZW5kcyA9IFtdO1xuICB9KTtcblxuICBpdCgncmVtb3ZlQmFja2VuZCBkaXNwb3NlcyB0aGUgYmFja2VuZCBhbmQgcmVtb3ZlcyB0aGUgZmFjdG9yeScsICgpID0+IHtcbiAgICBsZXQgYmFja2VuZDogS2VybmVsQmFja2VuZDtcbiAgICBjb25zdCBmYWN0b3J5ID0gKCkgPT4ge1xuICAgICAgY29uc3QgbmV3QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgICAgaWYgKGJhY2tlbmQgPT0gbnVsbCkge1xuICAgICAgICBiYWNrZW5kID0gbmV3QmFja2VuZDtcbiAgICAgICAgc3B5T24oYmFja2VuZCwgJ2Rpc3Bvc2UnKS5hbmQuY2FsbFRocm91Z2goKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBuZXdCYWNrZW5kO1xuICAgIH07XG5cbiAgICByZWdpc3RlckJhY2tlbmQoJ3Rlc3QtYmFja2VuZCcsIGZhY3RvcnkpO1xuXG4gICAgZXhwZWN0KHRmLmZpbmRCYWNrZW5kKCd0ZXN0LWJhY2tlbmQnKSAhPSBudWxsKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCh0Zi5maW5kQmFja2VuZCgndGVzdC1iYWNrZW5kJykpLnRvQmUoYmFja2VuZCk7XG4gICAgZXhwZWN0KHRmLmZpbmRCYWNrZW5kRmFjdG9yeSgndGVzdC1iYWNrZW5kJykpLnRvQmUoZmFjdG9yeSk7XG5cbiAgICB0Zi5yZW1vdmVCYWNrZW5kKCd0ZXN0LWJhY2tlbmQnKTtcblxuICAgIGV4cGVjdCh0Zi5maW5kQmFja2VuZCgndGVzdC1iYWNrZW5kJykgPT0gbnVsbCkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodGYuZmluZEJhY2tlbmQoJ3Rlc3QtYmFja2VuZCcpKS50b0JlKG51bGwpO1xuICAgIGV4cGVjdCgoYmFja2VuZC5kaXNwb3NlIGFzIGphc21pbmUuU3B5KS5jYWxscy5jb3VudCgpKS50b0JlKDEpO1xuICAgIGV4cGVjdCh0Zi5maW5kQmFja2VuZEZhY3RvcnkoJ3Rlc3QtYmFja2VuZCcpKS50b0JlKG51bGwpO1xuICB9KTtcblxuICBpdCgnZmluZEJhY2tlbmQgaW5pdGlhbGl6ZXMgdGhlIGJhY2tlbmQnLCAoKSA9PiB7XG4gICAgbGV0IGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQ7XG4gICAgY29uc3QgZmFjdG9yeSA9ICgpID0+IHtcbiAgICAgIGNvbnN0IG5ld0JhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICAgIGlmIChiYWNrZW5kID09IG51bGwpIHtcbiAgICAgICAgYmFja2VuZCA9IG5ld0JhY2tlbmQ7XG4gICAgICB9XG4gICAgICByZXR1cm4gbmV3QmFja2VuZDtcbiAgICB9O1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnY3VzdG9tLWNwdScsIGZhY3RvcnkpO1xuXG4gICAgZXhwZWN0KHRmLmZpbmRCYWNrZW5kKCdjdXN0b20tY3B1JykgIT0gbnVsbCkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodGYuZmluZEJhY2tlbmQoJ2N1c3RvbS1jcHUnKSkudG9CZShiYWNrZW5kKTtcbiAgICBleHBlY3QodGYuZmluZEJhY2tlbmRGYWN0b3J5KCdjdXN0b20tY3B1JykpLnRvQmUoZmFjdG9yeSk7XG4gIH0pO1xuXG4gIGl0KCdjdXN0b20gYmFja2VuZCByZWdpc3RyYXRpb24nLCAoKSA9PiB7XG4gICAgbGV0IGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQ7XG4gICAgY29uc3QgcHJpb3JpdHkgPSAxMDM7XG4gICAgcmVnaXN0ZXJCYWNrZW5kKCdjdXN0b20tY3B1JywgKCkgPT4ge1xuICAgICAgY29uc3QgbmV3QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgICAgaWYgKGJhY2tlbmQgPT0gbnVsbCkge1xuICAgICAgICBiYWNrZW5kID0gbmV3QmFja2VuZDtcbiAgICAgIH1cbiAgICAgIHJldHVybiBuZXdCYWNrZW5kO1xuICAgIH0sIHByaW9yaXR5KTtcblxuICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkgIT0gbnVsbCkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodGYuYmFja2VuZCgpKS50b0JlKGJhY2tlbmQpO1xuICB9KTtcblxuICBpdCgnaGlnaCBwcmlvcml0eSBiYWNrZW5kIHJlZ2lzdHJhdGlvbiBmYWlscywgZmFsbHMgYmFjaycsICgpID0+IHtcbiAgICBsZXQgbG93UHJpb3JpdHlCYWNrZW5kOiBLZXJuZWxCYWNrZW5kO1xuICAgIGNvbnN0IGxvd1ByaW9yaXR5ID0gMTAzO1xuICAgIGNvbnN0IGhpZ2hQcmlvcml0eSA9IDEwNDtcbiAgICByZWdpc3RlckJhY2tlbmQoJ2N1c3RvbS1sb3ctcHJpb3JpdHknLCAoKSA9PiB7XG4gICAgICBsb3dQcmlvcml0eUJhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICAgIHJldHVybiBsb3dQcmlvcml0eUJhY2tlbmQ7XG4gICAgfSwgbG93UHJpb3JpdHkpO1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnY3VzdG9tLWhpZ2gtcHJpb3JpdHknLCAoKSA9PiB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEhpZ2ggcHJpb3JpdHkgYmFja2VuZCBmYWlsc2ApO1xuICAgIH0sIGhpZ2hQcmlvcml0eSk7XG5cbiAgICBleHBlY3QodGYuYmFja2VuZCgpICE9IG51bGwpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHRmLmJhY2tlbmQoKSkudG9CZShsb3dQcmlvcml0eUJhY2tlbmQpO1xuICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvQmUoJ2N1c3RvbS1sb3ctcHJpb3JpdHknKTtcbiAgfSk7XG5cbiAgaXQoJ2xvdyBwcmlvcml0eSBhbmQgaGlnaCBwcmlvcml0eSBiYWNrZW5kcywgc2V0QmFja2VuZCBsb3cgcHJpb3JpdHknLCAoKSA9PiB7XG4gICAgbGV0IGxvd1ByaW9yaXR5QmFja2VuZDogS2VybmVsQmFja2VuZDtcbiAgICBsZXQgaGlnaFByaW9yaXR5QmFja2VuZDogS2VybmVsQmFja2VuZDtcbiAgICBjb25zdCBsb3dQcmlvcml0eSA9IDEwMztcbiAgICBjb25zdCBoaWdoUHJpb3JpdHkgPSAxMDQ7XG4gICAgcmVnaXN0ZXJCYWNrZW5kKCdjdXN0b20tbG93LXByaW9yaXR5JywgKCkgPT4ge1xuICAgICAgbG93UHJpb3JpdHlCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICByZXR1cm4gbG93UHJpb3JpdHlCYWNrZW5kO1xuICAgIH0sIGxvd1ByaW9yaXR5KTtcbiAgICByZWdpc3RlckJhY2tlbmQoJ2N1c3RvbS1oaWdoLXByaW9yaXR5JywgKCkgPT4ge1xuICAgICAgaGlnaFByaW9yaXR5QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgICAgcmV0dXJuIGhpZ2hQcmlvcml0eUJhY2tlbmQ7XG4gICAgfSwgaGlnaFByaW9yaXR5KTtcblxuICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkgIT0gbnVsbCkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodGYuYmFja2VuZCgpKS50b0JlKGhpZ2hQcmlvcml0eUJhY2tlbmQpO1xuICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvQmUoJ2N1c3RvbS1oaWdoLXByaW9yaXR5Jyk7XG5cbiAgICB0Zi5zZXRCYWNrZW5kKCdjdXN0b20tbG93LXByaW9yaXR5Jyk7XG5cbiAgICBleHBlY3QodGYuYmFja2VuZCgpICE9IG51bGwpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHRmLmJhY2tlbmQoKSkudG9CZShsb3dQcmlvcml0eUJhY2tlbmQpO1xuICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvQmUoJ2N1c3RvbS1sb3ctcHJpb3JpdHknKTtcbiAgfSk7XG5cbiAgaXQoJ2RlZmF1bHQgY3VzdG9tIGJhY2tncm91bmQgbnVsbCcsICgpID0+IHtcbiAgICBleHBlY3QodGYuZmluZEJhY2tlbmQoJ2N1c3RvbScpKS50b0JlTnVsbCgpO1xuICB9KTtcblxuICBpdCgnYWxsb3cgY3VzdG9tIGJhY2tlbmQnLCAoKSA9PiB7XG4gICAgY29uc3QgYmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgIGNvbnN0IHN1Y2Nlc3MgPSByZWdpc3RlckJhY2tlbmQoJ2N1c3RvbScsICgpID0+IGJhY2tlbmQpO1xuICAgIGV4cGVjdChzdWNjZXNzKS50b0JlVHJ1dGh5KCk7XG4gICAgZXhwZWN0KHRmLmZpbmRCYWNrZW5kKCdjdXN0b20nKSkudG9FcXVhbChiYWNrZW5kKTtcbiAgfSk7XG5cbiAgaXQoJ3N5bmMgYmFja2VuZCB3aXRoIGF3YWl0IHJlYWR5IHdvcmtzJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHRlc3RCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgcmVnaXN0ZXJCYWNrZW5kKCdzeW5jJywgKCkgPT4gdGVzdEJhY2tlbmQpO1xuICAgIHRmLnNldEJhY2tlbmQoJ3N5bmMnKTtcblxuICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvRXF1YWwoJ3N5bmMnKTtcbiAgICBhd2FpdCB0Zi5yZWFkeSgpO1xuICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkpLnRvRXF1YWwodGVzdEJhY2tlbmQpO1xuICB9KTtcblxuICBpdCgnc3luYyBiYWNrZW5kIHdpdGhvdXQgYXdhaXQgcmVhZHkgd29ya3MnLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgdGVzdEJhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICByZWdpc3RlckJhY2tlbmQoJ3N5bmMnLCAoKSA9PiB0ZXN0QmFja2VuZCk7XG4gICAgdGYuc2V0QmFja2VuZCgnc3luYycpO1xuXG4gICAgZXhwZWN0KHRmLmdldEJhY2tlbmQoKSkudG9FcXVhbCgnc3luYycpO1xuICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkpLnRvRXF1YWwodGVzdEJhY2tlbmQpO1xuICB9KTtcblxuICBpdCgnYXN5bmMgYmFja2VuZCB3aXRoIGF3YWl0IHJlYWR5IHdvcmtzJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHRlc3RCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgcmVnaXN0ZXJCYWNrZW5kKCdhc3luYycsIGFzeW5jICgpID0+IHtcbiAgICAgIGF3YWl0IHRmLm5leHRGcmFtZSgpO1xuICAgICAgcmV0dXJuIHRlc3RCYWNrZW5kO1xuICAgIH0pO1xuICAgIHRmLnNldEJhY2tlbmQoJ2FzeW5jJyk7XG5cbiAgICBleHBlY3QodGYuZ2V0QmFja2VuZCgpKS50b0VxdWFsKCdhc3luYycpO1xuICAgIGF3YWl0IHRmLnJlYWR5KCk7XG4gICAgZXhwZWN0KHRmLmJhY2tlbmQoKSkudG9FcXVhbCh0ZXN0QmFja2VuZCk7XG4gIH0pO1xuXG4gIGl0KCdhc3luYyBiYWNrZW5kIHdpdGhvdXQgYXdhaXQgcmVhZHkgZG9lcyBub3Qgd29yaycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCB0ZXN0QmFja2VuZCA9IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpO1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnYXN5bmMnLCBhc3luYyAoKSA9PiB7XG4gICAgICBhd2FpdCB0Zi5uZXh0RnJhbWUoKTtcbiAgICAgIHJldHVybiB0ZXN0QmFja2VuZDtcbiAgICB9KTtcbiAgICB0Zi5zZXRCYWNrZW5kKCdhc3luYycpO1xuXG4gICAgZXhwZWN0KHRmLmdldEJhY2tlbmQoKSkudG9FcXVhbCgnYXN5bmMnKTtcbiAgICBleHBlY3QoKCkgPT4gdGYuYmFja2VuZCgpKVxuICAgICAgICAudG9UaHJvd0Vycm9yKC9CYWNrZW5kICdhc3luYycgaGFzIG5vdCB5ZXQgYmVlbiBpbml0aWFsaXplZC4vKTtcbiAgfSk7XG5cbiAgaXQoJ3RmLnNxdWFyZSgpIGZhaWxzIGlmIHVzZXIgZG9lcyBub3QgYXdhaXQgcmVhZHkgb24gYXN5bmMgYmFja2VuZCcsXG4gICAgIGFzeW5jICgpID0+IHtcbiAgICAgICByZWdpc3RlckJhY2tlbmQoJ2FzeW5jJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICAgYXdhaXQgdGYubmV4dEZyYW1lKCk7XG4gICAgICAgICByZXR1cm4gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICAgfSk7XG4gICAgICAgdGYuc2V0QmFja2VuZCgnYXN5bmMnKTtcbiAgICAgICBleHBlY3QoKCkgPT4gdGYuc3F1YXJlKDIpKVxuICAgICAgICAgICAudG9UaHJvd0Vycm9yKC9CYWNrZW5kICdhc3luYycgaGFzIG5vdCB5ZXQgYmVlbiBpbml0aWFsaXplZC8pO1xuICAgICB9KTtcblxuICBpdCgndGYuc3F1YXJlKCkgd29ya3Mgd2hlbiB1c2VyIGF3YWl0cyByZWFkeSBvbiBhc3luYyBiYWNrZW5kJywgYXN5bmMgKCkgPT4ge1xuICAgIHJlZ2lzdGVyQmFja2VuZCgnYXN5bmMnLCBhc3luYyAoKSA9PiB7XG4gICAgICBhd2FpdCB0Zi5uZXh0RnJhbWUoKTtcbiAgICAgIHJldHVybiBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICB9KTtcbiAgICB0Zi5zZXRCYWNrZW5kKCdhc3luYycpO1xuICAgIGF3YWl0IHRmLnJlYWR5KCk7XG4gICAgZXhwZWN0KCgpID0+IHRmLnNxdWFyZSgyKSkudG9UaHJvd0Vycm9yKC8nd3JpdGUnIG5vdCB5ZXQgaW1wbGVtZW50ZWQvKTtcbiAgfSk7XG5cbiAgaXQoJ1JlZ2lzdGVyaW5nIGFzeW5jMiAoaGlnaGVyIHByaW9yaXR5KSBmYWlscywgYXN5bmMxIGJlY29tZXMgYWN0aXZlJyxcbiAgICAgYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IHRlc3RCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICAgcmVnaXN0ZXJCYWNrZW5kKCdhc3luYzEnLCBhc3luYyAoKSA9PiB7XG4gICAgICAgICBhd2FpdCB0Zi5uZXh0RnJhbWUoKTtcbiAgICAgICAgIHJldHVybiB0ZXN0QmFja2VuZDtcbiAgICAgICB9LCAxMDAgLyogcHJpb3JpdHkgKi8pO1xuICAgICAgIHJlZ2lzdGVyQmFja2VuZCgnYXN5bmMyJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICAgYXdhaXQgdGYubmV4dEZyYW1lKCk7XG4gICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ2ZhaWxlZCB0byBjcmVhdGUgYXN5bmMyJyk7XG4gICAgICAgfSwgMTAxIC8qIHByaW9yaXR5ICovKTtcblxuICAgICAgIC8vIEF3YWl0IGZvciB0aGUgbGlicmFyeSB0byBmaW5kIHRoZSBiZXN0IGJhY2tlbmQgdGhhdCBzdWNjZXNmdWxseVxuICAgICAgIC8vIGluaXRpYWxpemVzLlxuICAgICAgIGF3YWl0IHRmLnJlYWR5KCk7XG4gICAgICAgZXhwZWN0KHRmLmJhY2tlbmQoKSkudG9FcXVhbCh0ZXN0QmFja2VuZCk7XG4gICAgICAgZXhwZWN0KHRmLmdldEJhY2tlbmQoKSkudG9CZSgnYXN5bmMxJyk7XG4gICAgIH0pO1xuXG4gIGl0KCdSZWdpc3RlcmluZyBzeW5jIGFzIGhpZ2hlciBwcmlvcml0eSBhbmQgYXN5bmMgYXMgbG93ZXIgcHJpb3JpdHknLFxuICAgICBhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgdGVzdEJhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICAgICByZWdpc3RlckJhY2tlbmQoJ3N5bmMnLCAoKSA9PiB0ZXN0QmFja2VuZCwgMTAxIC8qIHByaW9yaXR5ICovKTtcbiAgICAgICByZWdpc3RlckJhY2tlbmQoJ2FzeW5jJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICAgYXdhaXQgdGYubmV4dEZyYW1lKCk7XG4gICAgICAgICByZXR1cm4gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICAgfSwgMTAwIC8qIHByaW9yaXR5ICovKTtcblxuICAgICAgIC8vIE5vIG5lZWQgdG8gYXdhaXQgZm9yIHJlYWR5KCkgc2luY2UgdGhlIGhpZ2hlc3QgcHJpb3JpdHkgb25lIGlzIHN5bmMuXG4gICAgICAgZXhwZWN0KHRmLmJhY2tlbmQoKSkudG9FcXVhbCh0ZXN0QmFja2VuZCk7XG4gICAgICAgZXhwZWN0KHRmLmdldEJhY2tlbmQoKSkudG9CZSgnc3luYycpO1xuICAgICB9KTtcblxuICBpdCgnYXN5bmMgYXMgaGlnaGVyIHByaW9yaXR5IGFuZCBzeW5jIGFzIGxvd2VyIHByaW9yaXR5IHdpdGggYXdhaXQgcmVhZHknLFxuICAgICBhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgdGVzdEJhY2tlbmQgPSBuZXcgVGVzdEtlcm5lbEJhY2tlbmQoKTtcbiAgICAgICByZWdpc3RlckJhY2tlbmQoJ2FzeW5jJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICAgYXdhaXQgdGYubmV4dEZyYW1lKCk7XG4gICAgICAgICByZXR1cm4gdGVzdEJhY2tlbmQ7XG4gICAgICAgfSwgMTAxIC8qIHByaW9yaXR5ICovKTtcbiAgICAgICByZWdpc3RlckJhY2tlbmQoXG4gICAgICAgICAgICdzeW5jJywgKCkgPT4gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCksIDEwMCAvKiBwcmlvcml0eSAqLyk7XG5cbiAgICAgICBhd2FpdCB0Zi5yZWFkeSgpO1xuICAgICAgIGV4cGVjdCh0Zi5iYWNrZW5kKCkpLnRvRXF1YWwodGVzdEJhY2tlbmQpO1xuICAgICAgIGV4cGVjdCh0Zi5nZXRCYWNrZW5kKCkpLnRvQmUoJ2FzeW5jJyk7XG4gICAgIH0pO1xuXG4gIGl0KCdhc3luYyBhcyBoaWdoZXIgcHJpb3JpdHkgYW5kIHN5bmMgYXMgbG93ZXIgcHJpb3JpdHkgdy9vIGF3YWl0IHJlYWR5JyxcbiAgICAgYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IHRlc3RCYWNrZW5kID0gbmV3IFRlc3RLZXJuZWxCYWNrZW5kKCk7XG4gICAgICAgcmVnaXN0ZXJCYWNrZW5kKCdhc3luYycsIGFzeW5jICgpID0+IHtcbiAgICAgICAgIGF3YWl0IHRmLm5leHRGcmFtZSgpO1xuICAgICAgICAgcmV0dXJuIHRlc3RCYWNrZW5kO1xuICAgICAgIH0sIDEwMSAvKiBwcmlvcml0eSAqLyk7XG4gICAgICAgcmVnaXN0ZXJCYWNrZW5kKFxuICAgICAgICAgICAnc3luYycsICgpID0+IG5ldyBUZXN0S2VybmVsQmFja2VuZCgpLCAxMDAgLyogcHJpb3JpdHkgKi8pO1xuXG4gICAgICAgZXhwZWN0KCgpID0+IHRmLmJhY2tlbmQoKSlcbiAgICAgICAgICAgLnRvVGhyb3dFcnJvcihcbiAgICAgICAgICAgICAgIC9UaGUgaGlnaGVzdCBwcmlvcml0eSBiYWNrZW5kICdhc3luYycgaGFzIG5vdCB5ZXQgYmVlbi8pO1xuICAgICB9KTtcblxuICBpdCgnUmVnaXN0ZXJpbmcgYW5kIHNldHRpbmcgYSBiYWNrZW5kIHRoYXQgZmFpbHMgdG8gcmVnaXN0ZXInLCBhc3luYyAoKSA9PiB7XG4gICAgcmVnaXN0ZXJCYWNrZW5kKCdhc3luYycsIGFzeW5jICgpID0+IHtcbiAgICAgIGF3YWl0IHRmLm5leHRGcmFtZSgpO1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdmYWlsZWQgdG8gY3JlYXRlIGFzeW5jJyk7XG4gICAgfSk7XG4gICAgY29uc3Qgc3VjY2VzcyA9IHRmLnNldEJhY2tlbmQoJ2FzeW5jJyk7XG4gICAgZXhwZWN0KHRmLmdldEJhY2tlbmQoKSkudG9CZSgnYXN5bmMnKTtcbiAgICBleHBlY3QoKCkgPT4gdGYuYmFja2VuZCgpKVxuICAgICAgICAudG9UaHJvd0Vycm9yKC9CYWNrZW5kICdhc3luYycgaGFzIG5vdCB5ZXQgYmVlbiBpbml0aWFsaXplZC8pO1xuICAgIGV4cGVjdChhd2FpdCBzdWNjZXNzKS50b0JlKGZhbHNlKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ21lbW9yeScsIEFMTF9FTlZTLCAoKSA9PiB7XG4gIGl0KCdTdW0oZmxvYXQpJywgYXN5bmMgKCkgPT4ge1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1CeXRlcykudG9CZSgwKTtcbiAgICBjb25zdCBzdW0gPSB0Zi50aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMiwgMywgNF0pO1xuICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMSk7XG4gICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtQnl0ZXMpLnRvQmUoNCAqIDQpO1xuICAgICAgcmV0dXJuIGEuc3VtKCk7XG4gICAgfSk7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMSk7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bUJ5dGVzKS50b0JlKDQpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHN1bS5kYXRhKCksIFsxICsgMiArIDMgKyA0XSk7XG4gIH0pO1xuXG4gIGl0KCdTdW0oYm9vbCknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc3VtID0gdGYudGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBhID0gdGYudGVuc29yMWQoW3RydWUsIHRydWUsIGZhbHNlLCB0cnVlXSwgJ2Jvb2wnKTtcbiAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDEpO1xuICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bUJ5dGVzKS50b0JlKDQpO1xuICAgICAgcmV0dXJuIGEuc3VtKCk7XG4gICAgfSk7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMSk7XG4gICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bUJ5dGVzKS50b0JlKDQpO1xuICAgIGV4cGVjdChzdW0uZHR5cGUpLnRvQmUoJ2ludDMyJyk7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgc3VtLmRhdGEoKSwgWzEgKyAxICsgMCArIDFdKTtcbiAgfSk7XG5cbiAgaXQoJ1N1bShpbnQzMiknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc3VtID0gdGYudGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCBhID0gdGYudGVuc29yMWQoWzEsIDEsIDAsIDFdLCAnaW50MzInKTtcbiAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDEpO1xuICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bUJ5dGVzKS50b0JlKDQgKiA0KTtcbiAgICAgIHJldHVybiBhLnN1bSgpO1xuICAgIH0pO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDEpO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1CeXRlcykudG9CZSg0KTtcbiAgICBleHBlY3Qoc3VtLmR0eXBlKS50b0JlKCdpbnQzMicpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHN1bS5kYXRhKCksIFsxICsgMSArIDAgKyAxXSk7XG4gIH0pO1xuXG4gIGl0KCdzdHJpbmcgdGVuc29yJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IoW1snYScsICdiYiddLCBbJ2MnLCAnZCddXSk7XG5cbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgxKTtcbiAgICBleHBlY3QodGYubWVtb3J5KCkubnVtQnl0ZXMpLnRvQmUoNSk7ICAvLyA1IGxldHRlcnMsIGVhY2ggMSBieXRlIGluIHV0ZjguXG5cbiAgICBhLmRpc3Bvc2UoKTtcblxuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1CeXRlcykudG9CZSgwKTtcbiAgfSk7XG5cbiAgaXQoJ3VucmVsaWFibGUgaXMgdHJ1ZSBmb3Igc3RyaW5nIHRlbnNvcnMnLCAoKSA9PiB7XG4gICAgdGYudGVuc29yKCdhJyk7XG4gICAgY29uc3QgbWVtID0gdGYubWVtb3J5KCk7XG4gICAgZXhwZWN0KG1lbS51bnJlbGlhYmxlKS50b0JlKHRydWUpO1xuICAgIGNvbnN0IGV4cGVjdGVkUmVhc29uID0gJ01lbW9yeSB1c2FnZSBieSBzdHJpbmcgdGVuc29ycyBpcyBhcHByb3hpbWF0ZSAnICtcbiAgICAgICAgJygyIGJ5dGVzIHBlciBjaGFyYWN0ZXIpJztcbiAgICBleHBlY3QobWVtLnJlYXNvbnMuaW5kZXhPZihleHBlY3RlZFJlYXNvbikgPj0gMCkudG9CZSh0cnVlKTtcbiAgfSk7XG5cbiAgaXQoJ21ha2VUZW5zb3JGcm9tRGF0YUlkIGNyZWF0ZXMgYSB0ZW5zb3InLCAoKSA9PiB7XG4gICAgY29uc3QgdGVuc29yID0gRU5HSU5FLm1ha2VUZW5zb3JGcm9tRGF0YUlkKHt9LCBbM10sICdmbG9hdDMyJyk7XG4gICAgZXhwZWN0KHRlbnNvcikudG9CZURlZmluZWQoKTtcbiAgICBleHBlY3QodGVuc29yLnNoYXBlKS50b0VxdWFsKFszXSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdwcm9maWxlJywgQUxMX0VOVlMsICgpID0+IHtcbiAgaXQoJ3NxdWFyaW5nJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHByb2ZpbGUgPSBhd2FpdCB0Zi5wcm9maWxlKCgpID0+IHtcbiAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICAgICAgbGV0IHgyID0geC5zcXVhcmUoKTtcbiAgICAgIHgyLmRpc3Bvc2UoKTtcbiAgICAgIHgyID0geC5zcXVhcmUoKTtcbiAgICAgIHgyLmRpc3Bvc2UoKTtcbiAgICAgIHJldHVybiB4O1xuICAgIH0pO1xuXG4gICAgY29uc3QgcmVzdWx0ID0gcHJvZmlsZS5yZXN1bHQgYXMgVGVuc29yO1xuXG4gICAgZXhwZWN0KHByb2ZpbGUubmV3Qnl0ZXMpLnRvQmUoMTIpO1xuICAgIGV4cGVjdChwcm9maWxlLnBlYWtCeXRlcykudG9CZSgyNCk7XG4gICAgZXhwZWN0KHByb2ZpbGUubmV3VGVuc29ycykudG9CZSgxKTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCByZXN1bHQuZGF0YSgpLCBbMSwgMiwgM10pO1xuICAgIGV4cGVjdChwcm9maWxlLmtlcm5lbHMubGVuZ3RoKS50b0JlKDIpO1xuXG4gICAgLy8gVGVzdCB0aGUgdHlwZXMgZm9yIGBrZXJuZWxUaW1lTXNgIGFuZCBgZXh0cmFJbmZvYCB0byBjb25maXJtIHRoZSBwcm9taXNlc1xuICAgIC8vIGFyZSByZXNvbHZlZC5cbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzBdLmtlcm5lbFRpbWVNcyBpbnN0YW5jZW9mIFByb21pc2UpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdChwcm9maWxlLmtlcm5lbHNbMF0uZXh0cmFJbmZvIGluc3RhbmNlb2YgUHJvbWlzZSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHByb2ZpbGUua2VybmVsc1sxXS5rZXJuZWxUaW1lTXMgaW5zdGFuY2VvZiBQcm9taXNlKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzFdLmV4dHJhSW5mbyBpbnN0YW5jZW9mIFByb21pc2UpLnRvQmUoZmFsc2UpO1xuXG4gICAgLy8gVGhlIHNwZWNpZmljIHZhbHVlcyBvZiBga2VybmVsVGltZU1zYCBhbmQgYGV4dHJhSW5mb2AgYXJlIHRlc3RlZCBpbiB0aGVcbiAgICAvLyB0ZXN0cyBvZiBQcm9maWxlci5wcm9maWxlS2VybmVsLCBzbyB0aGVpciB2YWx1ZXMgYXJlIG5vdCB0ZXN0ZWQgaGVyZS5cbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzBdKS50b0VxdWFsKHtcbiAgICAgICduYW1lJzogJ1NxdWFyZScsXG4gICAgICAnYnl0ZXNBZGRlZCc6IDEyLFxuICAgICAgJ3RvdGFsQnl0ZXNTbmFwc2hvdCc6IDI0LFxuICAgICAgJ3RlbnNvcnNBZGRlZCc6IDEsXG4gICAgICAndG90YWxUZW5zb3JzU25hcHNob3QnOiAyLFxuICAgICAgJ2lucHV0U2hhcGVzJzogW1szXV0sXG4gICAgICAnb3V0cHV0U2hhcGVzJzogW1szXV0sXG4gICAgICAna2VybmVsVGltZU1zJzogcHJvZmlsZS5rZXJuZWxzWzBdLmtlcm5lbFRpbWVNcyxcbiAgICAgICdleHRyYUluZm8nOiBwcm9maWxlLmtlcm5lbHNbMF0uZXh0cmFJbmZvXG4gICAgfSk7XG5cbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzFdKS50b0VxdWFsKHtcbiAgICAgICduYW1lJzogJ1NxdWFyZScsXG4gICAgICAnYnl0ZXNBZGRlZCc6IDEyLFxuICAgICAgJ3RvdGFsQnl0ZXNTbmFwc2hvdCc6IDI0LFxuICAgICAgJ3RlbnNvcnNBZGRlZCc6IDEsXG4gICAgICAndG90YWxUZW5zb3JzU25hcHNob3QnOiAyLFxuICAgICAgJ2lucHV0U2hhcGVzJzogW1szXV0sXG4gICAgICAnb3V0cHV0U2hhcGVzJzogW1szXV0sXG4gICAgICAna2VybmVsVGltZU1zJzogcHJvZmlsZS5rZXJuZWxzWzFdLmtlcm5lbFRpbWVNcyxcbiAgICAgICdleHRyYUluZm8nOiBwcm9maWxlLmtlcm5lbHNbMV0uZXh0cmFJbmZvXG4gICAgfSk7XG4gIH0pO1xuXG4gIGl0KCdzcXVhcmluZyB3aXRob3V0IGRpc3Bvc2luZycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBwcm9maWxlID0gYXdhaXQgdGYucHJvZmlsZSgoKSA9PiB7XG4gICAgICBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAgICAgIGNvbnN0IHgyID0geC5zcXVhcmUoKTtcbiAgICAgIHJldHVybiB4MjtcbiAgICB9KTtcblxuICAgIGNvbnN0IHJlc3VsdCA9IHByb2ZpbGUucmVzdWx0IGFzIFRlbnNvcjtcblxuICAgIGV4cGVjdChwcm9maWxlLm5ld0J5dGVzKS50b0JlKDI0KTtcbiAgICBleHBlY3QocHJvZmlsZS5wZWFrQnl0ZXMpLnRvQmUoMjQpO1xuICAgIGV4cGVjdChwcm9maWxlLm5ld1RlbnNvcnMpLnRvQmUoMik7XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgcmVzdWx0LmRhdGEoKSwgWzEsIDQsIDldKTtcbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzLmxlbmd0aCkudG9CZSgxKTtcbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzBdLmtlcm5lbFRpbWVNcyBpbnN0YW5jZW9mIFByb21pc2UpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdChwcm9maWxlLmtlcm5lbHNbMF0uZXh0cmFJbmZvIGluc3RhbmNlb2YgUHJvbWlzZSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHByb2ZpbGUua2VybmVsc1swXSkudG9FcXVhbCh7XG4gICAgICAnbmFtZSc6ICdTcXVhcmUnLFxuICAgICAgJ2J5dGVzQWRkZWQnOiAxMixcbiAgICAgICd0b3RhbEJ5dGVzU25hcHNob3QnOiAyNCxcbiAgICAgICd0ZW5zb3JzQWRkZWQnOiAxLFxuICAgICAgJ3RvdGFsVGVuc29yc1NuYXBzaG90JzogMixcbiAgICAgICdpbnB1dFNoYXBlcyc6IFtbM11dLFxuICAgICAgJ291dHB1dFNoYXBlcyc6IFtbM11dLFxuICAgICAgJ2tlcm5lbFRpbWVNcyc6IHByb2ZpbGUua2VybmVsc1swXS5rZXJuZWxUaW1lTXMsXG4gICAgICAnZXh0cmFJbmZvJzogcHJvZmlsZS5rZXJuZWxzWzBdLmV4dHJhSW5mb1xuICAgIH0pO1xuICB9KTtcblxuICBpdCgnc3F1YXJpbmcgaW4gYXN5bmMgcXVlcnknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgcHJvZmlsZSA9IGF3YWl0IHRmLnByb2ZpbGUoYXN5bmMgKCkgPT4ge1xuICAgICAgYXdhaXQgbmV3IFByb21pc2UocmVzb2x2ZSA9PiBzZXRUaW1lb3V0KHJlc29sdmUsIDEpKTtcbiAgICAgIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICAgICAgY29uc3QgeDIgPSB4LnNxdWFyZSgpO1xuICAgICAgeDIuZGlzcG9zZSgpO1xuICAgICAgcmV0dXJuIHg7XG4gICAgfSk7XG5cbiAgICBjb25zdCByZXN1bHQgPSBwcm9maWxlLnJlc3VsdCBhcyBUZW5zb3I7XG5cbiAgICBleHBlY3QocHJvZmlsZS5uZXdCeXRlcykudG9CZSgxMik7XG4gICAgZXhwZWN0KHByb2ZpbGUucGVha0J5dGVzKS50b0JlKDI0KTtcbiAgICBleHBlY3QocHJvZmlsZS5uZXdUZW5zb3JzKS50b0JlKDEpO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHJlc3VsdC5kYXRhKCksIFsxLCAyLCAzXSk7XG4gICAgZXhwZWN0KHByb2ZpbGUua2VybmVscy5sZW5ndGgpLnRvQmUoMSk7XG4gICAgZXhwZWN0KHByb2ZpbGUua2VybmVsc1swXS5rZXJuZWxUaW1lTXMgaW5zdGFuY2VvZiBQcm9taXNlKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxzWzBdLmV4dHJhSW5mbyBpbnN0YW5jZW9mIFByb21pc2UpLnRvQmUoZmFsc2UpO1xuICAgIGV4cGVjdChwcm9maWxlLmtlcm5lbHNbMF0pLnRvRXF1YWwoe1xuICAgICAgJ25hbWUnOiAnU3F1YXJlJyxcbiAgICAgICdieXRlc0FkZGVkJzogMTIsXG4gICAgICAndG90YWxCeXRlc1NuYXBzaG90JzogMjQsXG4gICAgICAndGVuc29yc0FkZGVkJzogMSxcbiAgICAgICd0b3RhbFRlbnNvcnNTbmFwc2hvdCc6IDIsXG4gICAgICAnaW5wdXRTaGFwZXMnOiBbWzNdXSxcbiAgICAgICdvdXRwdXRTaGFwZXMnOiBbWzNdXSxcbiAgICAgICdrZXJuZWxUaW1lTXMnOiBwcm9maWxlLmtlcm5lbHNbMF0ua2VybmVsVGltZU1zLFxuICAgICAgJ2V4dHJhSW5mbyc6IHByb2ZpbGUua2VybmVsc1swXS5leHRyYUluZm9cbiAgICB9KTtcbiAgfSk7XG5cbiAgaXQoJ3JlcG9ydHMgY29ycmVjdCBrZXJuZWxOYW1lcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBwcm9maWxlID0gYXdhaXQgdGYucHJvZmlsZSgoKSA9PiB7XG4gICAgICBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAgICAgIGNvbnN0IHgyID0geC5zcXVhcmUoKTtcbiAgICAgIGNvbnN0IHgzID0geDIuYWJzKCk7XG4gICAgICByZXR1cm4geDM7XG4gICAgfSk7XG5cbiAgICBleHBlY3QocHJvZmlsZS5rZXJuZWxOYW1lcykudG9FcXVhbChqYXNtaW5lLmFycmF5V2l0aEV4YWN0Q29udGVudHMoW1xuICAgICAgJ1NxdWFyZScsICdBYnMnXG4gICAgXSkpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZVdpdGhGbGFncygnZGlzcG9zZVZhcmlhYmxlcycsIEFMTF9FTlZTLCAoKSA9PiB7XG4gIGl0KCdyZXVzZSBzYW1lIG5hbWUgdmFyaWFibGUnLCAoKSA9PiB7XG4gICAgdGYudGVuc29yMWQoWzEsIDIsIDNdKS52YXJpYWJsZSh0cnVlLCAndjEnKTtcbiAgICB0Zi50ZW5zb3IxZChbMSwgMiwgM10pLnZhcmlhYmxlKHRydWUsICd2MicpO1xuICAgIGV4cGVjdCgoKSA9PiB7XG4gICAgICB0Zi50ZW5zb3IxZChbMSwgMiwgM10pLnZhcmlhYmxlKHRydWUsICd2MScpO1xuICAgIH0pLnRvVGhyb3dFcnJvcigpO1xuICAgIHRmLmRpc3Bvc2VWYXJpYWJsZXMoKTtcbiAgICB0Zi50ZW5zb3IxZChbMSwgMiwgM10pLnZhcmlhYmxlKHRydWUsICd2MScpO1xuICAgIHRmLnRlbnNvcjFkKFsxLCAyLCAzXSkudmFyaWFibGUodHJ1ZSwgJ3YyJyk7XG4gIH0pO1xufSk7XG5cbi8qKlxuICogVGhlIGZvbGxvd2luZyB0ZXN0IGNvbnN0cmFpbnRzIHRvIHRoZSBDUFUgZW52aXJvbm1lbnQgYmVjYXVzZSBpdCBuZWVkcyBhXG4gKiBjb25jcmV0ZSBiYWNrZW5kIHRvIGV4aXN0LiBUaGlzIHRlc3Qgd2lsbCB3b3JrIGZvciBhbnkgYmFja2VuZCwgYnV0IGN1cnJlbnRseVxuICogdGhpcyBpcyB0aGUgc2ltcGxlc3QgYmFja2VuZCB0byB0ZXN0IGFnYWluc3QuXG4gKi9cbmRlc2NyaWJlV2l0aEZsYWdzKFxuICAgICdTd2l0Y2hpbmcgY3B1IGJhY2tlbmRzJyxcbiAgICB7cHJlZGljYXRlOiB0ZXN0RW52ID0+IHRlc3RFbnYuYmFja2VuZE5hbWUgPT09ICdjcHUnfSwgKCkgPT4ge1xuICAgICAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgICAgIHRmLnJlZ2lzdGVyQmFja2VuZCgnY3B1MScsIHRmLmZpbmRCYWNrZW5kRmFjdG9yeSgnY3B1JykpO1xuICAgICAgICB0Zi5yZWdpc3RlckJhY2tlbmQoJ2NwdTInLCB0Zi5maW5kQmFja2VuZEZhY3RvcnkoJ2NwdScpKTtcbiAgICAgIH0pO1xuXG4gICAgICBhZnRlckVhY2goKCkgPT4ge1xuICAgICAgICB0Zi5yZW1vdmVCYWNrZW5kKCdjcHUxJyk7XG4gICAgICAgIHRmLnJlbW92ZUJhY2tlbmQoJ2NwdTInKTtcbiAgICAgIH0pO1xuXG4gICAgICBpdCgnTW92ZSBkYXRhIGZyb20gY3B1MSB0byBjcHUyIGJhY2tlbmQnLCBhc3luYyAoKSA9PiB7XG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ2NwdTEnKTtcbiAgICAgICAgLy8gVGhpcyBzY2FsYXIgbGl2ZXMgaW4gY3B1MS5cbiAgICAgICAgY29uc3QgYSA9IHRmLnNjYWxhcig1KTtcblxuICAgICAgICB0Zi5zZXRCYWNrZW5kKCdjcHUyJyk7XG4gICAgICAgIC8vIFRoaXMgc2NhbGFyIGxpdmVzIGluIGNwdTIuXG4gICAgICAgIGNvbnN0IGIgPSB0Zi5zY2FsYXIoMyk7XG5cbiAgICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bURhdGFCdWZmZXJzKS50b0JlKDIpO1xuICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSgyKTtcbiAgICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bUJ5dGVzKS50b0JlKDgpO1xuXG4gICAgICAgIC8vIE1ha2Ugc3VyZSB5b3UgY2FuIHJlYWQgYm90aCB0ZW5zb3JzLlxuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCBhLmRhdGEoKSwgWzVdKTtcbiAgICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgYi5kYXRhKCksIFszXSk7XG5cbiAgICAgICAgLy8gU3dpdGNoIGJhY2sgdG8gY3B1MS5cbiAgICAgICAgdGYuc2V0QmFja2VuZCgnY3B1MScpO1xuICAgICAgICAvLyBBZ2FpbiBtYWtlIHN1cmUgeW91IGNhbiByZWFkIGJvdGggdGVuc29ycy5cbiAgICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgYS5kYXRhKCksIFs1XSk7XG4gICAgICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGIuZGF0YSgpLCBbM10pO1xuXG4gICAgICAgIHRmLmRpc3Bvc2UoW2EsIGJdKTtcblxuICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtRGF0YUJ1ZmZlcnMpLnRvQmUoMCk7XG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtQnl0ZXMpLnRvQmUoMCk7XG4gICAgICB9KTtcblxuICAgICAgaXQoJ2NhbiBleGVjdXRlIG9wIHdpdGggZGF0YSBmcm9tIG1peGVkIGJhY2tlbmRzJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICBjb25zdCBrZXJuZWxGdW5jID0gdGYuZ2V0S2VybmVsKCdBZGQnLCAnY3B1Jykua2VybmVsRnVuYztcbiAgICAgICAgdGYucmVnaXN0ZXJLZXJuZWwoe2tlcm5lbE5hbWU6ICdBZGQnLCBiYWNrZW5kTmFtZTogJ2NwdTEnLCBrZXJuZWxGdW5jfSk7XG4gICAgICAgIHRmLnJlZ2lzdGVyS2VybmVsKHtrZXJuZWxOYW1lOiAnQWRkJywgYmFja2VuZE5hbWU6ICdjcHUyJywga2VybmVsRnVuY30pO1xuXG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ2NwdTEnKTtcbiAgICAgICAgLy8gVGhpcyBzY2FsYXIgbGl2ZXMgaW4gY3B1MS5cbiAgICAgICAgY29uc3QgYSA9IHRmLnNjYWxhcig1KTtcblxuICAgICAgICB0Zi5zZXRCYWNrZW5kKCdjcHUyJyk7XG4gICAgICAgIC8vIFRoaXMgc2NhbGFyIGxpdmVzIGluIGNwdTIuXG4gICAgICAgIGNvbnN0IGIgPSB0Zi5zY2FsYXIoMyk7XG5cbiAgICAgICAgLy8gVmVyaWZ5IHRoYXQgb3BzIGNhbiBleGVjdXRlIHdpdGggbWl4ZWQgYmFja2VuZCBkYXRhLlxuICAgICAgICBFTkdJTkUuc3RhcnRTY29wZSgpO1xuICAgICAgICB0Zi5zZXRCYWNrZW5kKCdjcHUxJyk7XG4gICAgICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHRmLmFkZChhLCBiKS5kYXRhKCksIFs4XSk7XG5cbiAgICAgICAgdGYuc2V0QmFja2VuZCgnY3B1MicpO1xuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB0Zi5hZGQoYSwgYikuZGF0YSgpLCBbOF0pO1xuICAgICAgICBFTkdJTkUuZW5kU2NvcGUoKTtcblxuICAgICAgICB0Zi5kaXNwb3NlKFthLCBiXSk7XG4gICAgICB9KTtcbiAgICB9KTtcblxuLyoqXG4gKiBUaGUgZm9sbG93aW5nIHVuaXQgdGVzdCBpcyBhIHNwZWNpYWwgaW50ZWdyYXRpb24tc3R5bGUgdGVzdCB0aGF0IGFzc3VtZXNcbiAqIHRoaW5ncyBhYm91dCBDUFUgJiBXZWJHTCBiYWNrZW5kcyBiZWluZyByZWdpc3RlcmVkLiBUaGlzIHRlc3RzIGRvZXNuJ3QgbGl2ZVxuICogaW4gdGhlIGJhY2tlbmQgZGlyZWN0b3J5IGJlY2F1c2UgaXQgaXMgdGVzdGluZyBlbmdpbmUgcmF0aGVyIHRoYW5cbiAqIGJhY2tlbmQtc3BlY2lmaWMgZGV0YWlscyBidXQgbmVlZHMgYSByZWFsIGJhY2tlbmQgdG8gZXhpc3QuIFRoaXMgdGVzdCB3aWxsXG4gKiBmYWlsIGlmIHRoZSBDUFUgYmFja2VuZHMgaXMgbm90IHJlZ2lzdGVyZWQuIFRoaXMgaXMgaW50ZW50aW9uYWwsIHdlIHNob3VsZFxuICogaGF2ZSBjb3ZlcmFnZSBmb3Igd2hlbiB0aGVzZSBiYWNrZW5kcyBhcmUgZW5hYmxlZCBhbmQgZW5zdXJlIHRoZXkgd29yayB3aXRoXG4gKiB0aGUgZW5naW5lLlxuICovXG4vLyBUT0RPKCM1NjMyKTogUmUtZW5hYmxlIHRoZXNlIHRlc3RzXG4vKlxuZGVzY3JpYmVXaXRoRmxhZ3MoXG4gICAgJ1N3aXRjaGluZyBXZWJHTCArIENQVSBiYWNrZW5kcycsIHtcbiAgICAgIHByZWRpY2F0ZTogdGVzdEVudiA9PiB0ZXN0RW52LmJhY2tlbmROYW1lID09PSAnd2ViZ2wnICYmXG4gICAgICAgICAgRU5HSU5FLmJhY2tlbmROYW1lcygpLmluZGV4T2YoJ3dlYmdsJykgIT09IC0xICYmXG4gICAgICAgICAgRU5HSU5FLmJhY2tlbmROYW1lcygpLmluZGV4T2YoJ2NwdScpICE9PSAtMVxuICAgIH0sXG4gICAgKCkgPT4ge1xuICAgICAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgICAgIHRmLnJlZ2lzdGVyQmFja2VuZCgnd2ViZ2wxJywgdGYuZmluZEJhY2tlbmRGYWN0b3J5KCd3ZWJnbCcpKTtcbiAgICAgICAgdGYucmVnaXN0ZXJCYWNrZW5kKCd3ZWJnbDInLCB0Zi5maW5kQmFja2VuZEZhY3RvcnkoJ3dlYmdsJykpO1xuICAgICAgICB0Zi5yZWdpc3RlckJhY2tlbmQoJ2NwdTEnLCB0Zi5maW5kQmFja2VuZEZhY3RvcnkoJ2NwdScpKTtcbiAgICAgIH0pO1xuXG4gICAgICBhZnRlckVhY2goKCkgPT4ge1xuICAgICAgICB0Zi5yZW1vdmVCYWNrZW5kKCd3ZWJnbDEnKTtcbiAgICAgICAgdGYucmVtb3ZlQmFja2VuZCgnd2ViZ2wyJyk7XG4gICAgICAgIHRmLnJlbW92ZUJhY2tlbmQoJ2NwdTEnKTtcbiAgICAgIH0pO1xuXG4gICAgICBpdCgnY2FuIGV4ZWN1dGUgb3Agd2l0aCBkYXRhIGZyb20gbWl4ZWQgYmFja2VuZHMnLCBhc3luYyAoKSA9PiB7XG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ3dlYmdsMScpO1xuICAgICAgICBjb25zdCBhID0gdGYuc2NhbGFyKDUpO1xuXG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ3dlYmdsMicpO1xuICAgICAgICBjb25zdCBiID0gdGYuc2NhbGFyKDMpO1xuXG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ2NwdTEnKTtcbiAgICAgICAgY29uc3QgYyA9IHRmLnNjYWxhcigyKTtcblxuICAgICAgICAvLyBWZXJpZnkgdGhhdCBvcHMgY2FuIGV4ZWN1dGUgd2l0aCBtaXhlZCBiYWNrZW5kIGRhdGEuXG4gICAgICAgIEVOR0lORS5zdGFydFNjb3BlKCk7XG4gICAgICAgIHRmLnNldEJhY2tlbmQoJ3dlYmdsMScpO1xuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB0Zi5hZGROKFthLCBiLCBjXSkuZGF0YSgpLCBbMTBdKTtcblxuICAgICAgICB0Zi5zZXRCYWNrZW5kKCd3ZWJnbDInKTtcbiAgICAgICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgdGYuYWRkTihbYSwgYiwgY10pLmRhdGEoKSwgWzEwXSk7XG5cbiAgICAgICAgdGYuc2V0QmFja2VuZCgnY3B1MScpO1xuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB0Zi5hZGROKFthLCBiLCBjXSkuZGF0YSgpLCBbMTBdKTtcbiAgICAgICAgRU5HSU5FLmVuZFNjb3BlKCk7XG5cbiAgICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMyk7XG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1EYXRhQnVmZmVycykudG9CZSgzKTtcblxuICAgICAgICB0Zi5kaXNwb3NlKFthLCBiLCBjXSk7XG5cbiAgICAgICAgZXhwZWN0KHRmLm1lbW9yeSgpLm51bVRlbnNvcnMpLnRvQmUoMCk7XG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1EYXRhQnVmZmVycykudG9CZSgwKTtcbiAgICAgIH0pO1xuXG4gICAgICBpdCgnZnJvbVBpeGVscyB3aXRoIG1peGVkIGJhY2tlbmRzIHdvcmtzJywgYXN5bmMgKCkgPT4ge1xuICAgICAgICB0Zi5zZXRCYWNrZW5kKCd3ZWJnbDEnKTtcbiAgICAgICAgY29uc3QgYSA9IHRmLmJyb3dzZXIuZnJvbVBpeGVscyhcbiAgICAgICAgICAgIG5ldyBJbWFnZURhdGEobmV3IFVpbnQ4Q2xhbXBlZEFycmF5KFsxLCAyLCAzLCA0XSksIDEsIDEpKTtcblxuICAgICAgICB0Zi5zZXRCYWNrZW5kKCd3ZWJnbDInKTtcbiAgICAgICAgY29uc3QgYiA9IHRmLmJyb3dzZXIuZnJvbVBpeGVscyhcbiAgICAgICAgICAgIG5ldyBJbWFnZURhdGEobmV3IFVpbnQ4Q2xhbXBlZEFycmF5KFs1LCA2LCA3LCA4XSksIDEsIDEpKTtcblxuICAgICAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB0Zi5hZGQoYSwgYikuZGF0YSgpLCBbNiwgOCwgMTBdKTtcbiAgICAgIH0pO1xuXG4gICAgICBpdCgnc2luZ2xlIHRpZHkgbXVsdGlwbGUgYmFja2VuZHMnLCAoKSA9PiB7XG4gICAgICAgIGNvbnN0IGtlcm5lbEZ1bmMgPSB0Zi5nZXRLZXJuZWwoJ1NxdWFyZScsICd3ZWJnbCcpLmtlcm5lbEZ1bmM7XG4gICAgICAgIHRmLnJlZ2lzdGVyS2VybmVsKFxuICAgICAgICAgICAge2tlcm5lbE5hbWU6ICdTcXVhcmUnLCBiYWNrZW5kTmFtZTogJ3dlYmdsMScsIGtlcm5lbEZ1bmN9KTtcbiAgICAgICAgdGYucmVnaXN0ZXJLZXJuZWwoXG4gICAgICAgICAgICB7a2VybmVsTmFtZTogJ1NxdWFyZScsIGJhY2tlbmROYW1lOiAnd2ViZ2wyJywga2VybmVsRnVuY30pO1xuXG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuXG4gICAgICAgIHRmLnRpZHkoKCkgPT4ge1xuICAgICAgICAgIHRmLnNldEJhY2tlbmQoJ3dlYmdsMScpO1xuICAgICAgICAgIGNvbnN0IGEgPSB0Zi5zY2FsYXIoMSk7XG4gICAgICAgICAgYS5zcXVhcmUoKTsgIC8vIFVwbG9hZHMgdG8gR1BVLlxuXG4gICAgICAgICAgdGYuc2V0QmFja2VuZCgnd2ViZ2wyJyk7XG4gICAgICAgICAgY29uc3QgYiA9IHRmLnNjYWxhcigxKTtcbiAgICAgICAgICBiLnNxdWFyZSgpOyAgLy8gVXBsb2FkcyB0byBHUFUuXG5cbiAgICAgICAgICBleHBlY3QodGYubWVtb3J5KCkubnVtVGVuc29ycykudG9CZSg0KTtcbiAgICAgICAgfSk7XG4gICAgICAgIGV4cGVjdCh0Zi5tZW1vcnkoKS5udW1UZW5zb3JzKS50b0JlKDApO1xuXG4gICAgICAgIHRmLnVucmVnaXN0ZXJLZXJuZWwoJ1NxdWFyZScsICd3ZWJnbDEnKTtcbiAgICAgICAgdGYudW5yZWdpc3Rlcktlcm5lbCgnU3F1YXJlJywgJ3dlYmdsMicpO1xuICAgICAgfSk7XG4gICAgfSk7XG4qL1xuaW50ZXJmYWNlIFRlc3RTdG9yYWdlIGV4dGVuZHMgS2VybmVsQmFja2VuZCB7XG4gIGlkOiBudW1iZXI7XG59XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdEZXRlY3RzIG1lbW9yeSBsZWFrcyBpbiBrZXJuZWxzJywgQUxMX0VOVlMsICgpID0+IHtcbiAgY29uc3QgYmFja2VuZE5hbWUgPSAndGVzdC1tZW0nO1xuICBjb25zdCBrZXJuZWxOYW1lID0gJ015S2VybmVsJztcbiAgY29uc3Qga2VybmVsTmFtZUNvbXBsZXggPSAnS2VybmVsLWNvbXBsZXgnO1xuXG4gIGl0KCdEZXRlY3RzIG1lbW9yeSBsZWFrIGluIGEga2VybmVsJywgKCkgPT4ge1xuICAgIGxldCBkYXRhSWRzQ291bnQgPSAwO1xuICAgIHRmLnJlZ2lzdGVyQmFja2VuZChiYWNrZW5kTmFtZSwgKCkgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaWQ6IDEsXG4gICAgICAgIGRpc3Bvc2U6ICgpID0+IG51bGwsXG4gICAgICAgIGRpc3Bvc2VEYXRhOiAoZGF0YUlkOiB7fSkgPT4gbnVsbCxcbiAgICAgICAgbnVtRGF0YUlkczogKCkgPT4gZGF0YUlkc0NvdW50XG4gICAgICB9IGFzIFRlc3RTdG9yYWdlO1xuICAgIH0pO1xuXG4gICAgY29uc3Qga2VybmVsV2l0aE1lbUxlYWs6IEtlcm5lbEZ1bmMgPSAoKSA9PiB7XG4gICAgICBkYXRhSWRzQ291bnQgKz0gMjtcbiAgICAgIHJldHVybiB7ZGF0YUlkOiB7fSwgc2hhcGU6IFtdLCBkdHlwZTogJ2Zsb2F0MzInfTtcbiAgICB9O1xuICAgIHRmLnJlZ2lzdGVyS2VybmVsKHtrZXJuZWxOYW1lLCBiYWNrZW5kTmFtZSwga2VybmVsRnVuYzoga2VybmVsV2l0aE1lbUxlYWt9KTtcblxuICAgIHRmLnNldEJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgIGV4cGVjdCgoKSA9PiB0Zi5lbmdpbmUoKS5ydW5LZXJuZWwoa2VybmVsTmFtZSwge30sIHt9KSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcihcbiAgICAgICAgICAgIC9CYWNrZW5kICd0ZXN0LW1lbScgaGFzIGFuIGludGVybmFsIG1lbW9yeSBsZWFrIFxcKDEgZGF0YSBpZHNcXCkvKTtcblxuICAgIHRmLnJlbW92ZUJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgIHRmLnVucmVnaXN0ZXJLZXJuZWwoa2VybmVsTmFtZSwgYmFja2VuZE5hbWUpO1xuICB9KTtcblxuICBpdCgnTm8gbWVtIGxlYWsgaW4gYSBrZXJuZWwgd2l0aCBtdWx0aXBsZSBvdXRwdXRzJywgKCkgPT4ge1xuICAgIGxldCBkYXRhSWRzQ291bnQgPSAwO1xuICAgIHRmLnJlZ2lzdGVyQmFja2VuZChiYWNrZW5kTmFtZSwgKCkgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaWQ6IDEsXG4gICAgICAgIGRpc3Bvc2U6ICgpID0+IG51bGwsXG4gICAgICAgIGRpc3Bvc2VEYXRhOiAoZGF0YUlkOiB7fSkgPT4gbnVsbCxcbiAgICAgICAgbnVtRGF0YUlkczogKCkgPT4gZGF0YUlkc0NvdW50XG4gICAgICB9IGFzIFRlc3RTdG9yYWdlO1xuICAgIH0pO1xuICAgIHRmLnNldEJhY2tlbmQoYmFja2VuZE5hbWUpO1xuXG4gICAgY29uc3Qga2VybmVsV2l0aDNPdXRwdXRzOiBLZXJuZWxGdW5jID0gKCkgPT4ge1xuICAgICAgZGF0YUlkc0NvdW50ICs9IDM7XG4gICAgICBjb25zdCB0OiBUZW5zb3JJbmZvID0ge2RhdGFJZDoge30sIHNoYXBlOiBbXSwgZHR5cGU6ICdmbG9hdDMyJ307XG4gICAgICByZXR1cm4gW3QsIHQsIHRdO1xuICAgIH07XG4gICAgdGYucmVnaXN0ZXJLZXJuZWwoXG4gICAgICAgIHtrZXJuZWxOYW1lLCBiYWNrZW5kTmFtZSwga2VybmVsRnVuYzoga2VybmVsV2l0aDNPdXRwdXRzfSk7XG5cbiAgICBjb25zdCByZXMgPSB0Zi5lbmdpbmUoKS5ydW5LZXJuZWwoa2VybmVsTmFtZSwge30sIHt9KTtcbiAgICBleHBlY3QoQXJyYXkuaXNBcnJheShyZXMpKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCgocmVzIGFzIEFycmF5PHt9PikubGVuZ3RoKS50b0JlKDMpO1xuXG4gICAgY29uc3Qga2VybmVsV2l0aENvbXBsZXhPdXRwdXRzOiBLZXJuZWxGdW5jID0gKCkgPT4ge1xuICAgICAgZGF0YUlkc0NvdW50ICs9IDM7XG4gICAgICByZXR1cm4ge2RhdGFJZDoge30sIHNoYXBlOiBbXSwgZHR5cGU6ICdjb21wbGV4NjQnfTtcbiAgICB9O1xuICAgIHRmLnJlZ2lzdGVyS2VybmVsKHtcbiAgICAgIGtlcm5lbE5hbWU6IGtlcm5lbE5hbWVDb21wbGV4LFxuICAgICAgYmFja2VuZE5hbWUsXG4gICAgICBrZXJuZWxGdW5jOiBrZXJuZWxXaXRoQ29tcGxleE91dHB1dHNcbiAgICB9KTtcblxuICAgIGNvbnN0IHJlczIgPSB0Zi5lbmdpbmUoKS5ydW5LZXJuZWwoa2VybmVsTmFtZUNvbXBsZXgsIHt9LCB7fSkgYXMgVGVuc29ySW5mbztcbiAgICBleHBlY3QocmVzMi5zaGFwZSkudG9FcXVhbChbXSk7XG4gICAgZXhwZWN0KHJlczIuZHR5cGUpLnRvRXF1YWwoJ2NvbXBsZXg2NCcpO1xuXG4gICAgdGYucmVtb3ZlQmFja2VuZChiYWNrZW5kTmFtZSk7XG4gICAgdGYudW5yZWdpc3Rlcktlcm5lbChrZXJuZWxOYW1lLCBiYWNrZW5kTmFtZSk7XG4gICAgdGYudW5yZWdpc3Rlcktlcm5lbChrZXJuZWxOYW1lQ29tcGxleCwgYmFja2VuZE5hbWUpO1xuICB9KTtcbn0pO1xuXG4vLyBOT1RFOiBUaGlzIGRlc2NyaWJlIGlzIHB1cnBvc2VmdWxseSBub3QgYSBkZXNjcmliZVdpdGhGbGFncyBzbyB0aGF0IHdlXG4vLyB0ZXN0IHRlbnNvciBhbGxvY2F0aW9uIHdoZXJlIG5vIHNjb3BlcyBoYXZlIGJlZW4gY3JlYXRlZC5cbmRlc2NyaWJlKCdNZW1vcnkgYWxsb2NhdGlvbiBvdXRzaWRlIGEgdGVzdCBzY29wZScsICgpID0+IHtcbiAgaXQoJ2NvbnN0cnVjdGluZyBhIHRlbnNvciB3b3JrcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBiYWNrZW5kTmFtZSA9ICd0ZXN0LWJhY2tlbmQnO1xuICAgIHRmLnJlZ2lzdGVyQmFja2VuZChiYWNrZW5kTmFtZSwgKCkgPT4ge1xuICAgICAgbGV0IHN0b3JlZFZhbHVlczogQmFja2VuZFZhbHVlcyA9IG51bGw7XG4gICAgICByZXR1cm4ge1xuICAgICAgICBpZDogMSxcbiAgICAgICAgZmxvYXRQcmVjaXNpb246ICgpID0+IDMyLFxuICAgICAgICB3cml0ZTogKHZhbHVlczogQmFja2VuZFZhbHVlcywgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUpID0+IHtcbiAgICAgICAgICBjb25zdCBkYXRhSWQgPSB7fTtcbiAgICAgICAgICBzdG9yZWRWYWx1ZXMgPSB2YWx1ZXM7XG4gICAgICAgICAgcmV0dXJuIGRhdGFJZDtcbiAgICAgICAgfSxcbiAgICAgICAgcmVhZDogYXN5bmMgKGRhdGFJZDogb2JqZWN0KSA9PiBzdG9yZWRWYWx1ZXMsXG4gICAgICAgIGRpc3Bvc2U6ICgpID0+IG51bGwsXG4gICAgICAgIGRpc3Bvc2VEYXRhOiAoZGF0YUlkOiB7fSkgPT4gbnVsbFxuICAgICAgfSBhcyBUZXN0U3RvcmFnZTtcbiAgICB9KTtcbiAgICB0Zi5zZXRCYWNrZW5kKGJhY2tlbmROYW1lKTtcblxuICAgIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMiwgM10pO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IGEuZGF0YSgpLCBbMSwgMiwgM10pO1xuICAgIGEuZGlzcG9zZSgpO1xuXG4gICAgdGYucmVtb3ZlQmFja2VuZChiYWNrZW5kTmFtZSk7XG4gIH0pO1xufSk7XG4iXX0=