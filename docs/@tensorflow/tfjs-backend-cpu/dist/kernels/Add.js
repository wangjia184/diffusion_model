/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { Add } from '@tensorflow/tfjs-core';
import { createSimpleBinaryKernelImpl } from '../utils/binary_impl';
import { binaryKernelFunc, createComplexBinaryKernelImpl } from '../utils/binary_utils';
export const addImpl = createSimpleBinaryKernelImpl(((a, b) => a + b));
export const addComplexImpl = createComplexBinaryKernelImpl(((aReal, aImag, bReal, bImag) => {
    return { real: aReal + bReal, imag: aImag + bImag };
}));
export const add = binaryKernelFunc(Add, addImpl, addComplexImpl);
export const addConfig = {
    kernelName: Add,
    backendName: 'cpu',
    kernelFunc: add
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQWRkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLWNwdS9zcmMva2VybmVscy9BZGQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLEdBQUcsRUFBZSxNQUFNLHVCQUF1QixDQUFDO0FBRXhELE9BQU8sRUFBQyw0QkFBNEIsRUFBQyxNQUFNLHNCQUFzQixDQUFDO0FBQ2xFLE9BQU8sRUFBQyxnQkFBZ0IsRUFBRSw2QkFBNkIsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXRGLE1BQU0sQ0FBQyxNQUFNLE9BQU8sR0FDaEIsNEJBQTRCLENBQUMsQ0FBQyxDQUFDLENBQVMsRUFBRSxDQUFTLEVBQUUsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQ3BFLE1BQU0sQ0FBQyxNQUFNLGNBQWMsR0FDdkIsNkJBQTZCLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFFLEtBQUssRUFBRSxFQUFFO0lBQzVELE9BQU8sRUFBQyxJQUFJLEVBQUUsS0FBSyxHQUFHLEtBQUssRUFBRSxJQUFJLEVBQUUsS0FBSyxHQUFHLEtBQUssRUFBQyxDQUFDO0FBQ3BELENBQUMsQ0FBQyxDQUFDLENBQUM7QUFFUixNQUFNLENBQUMsTUFBTSxHQUFHLEdBQUcsZ0JBQWdCLENBQUMsR0FBRyxFQUFFLE9BQU8sRUFBRSxjQUFjLENBQUMsQ0FBQztBQUVsRSxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQWlCO0lBQ3JDLFVBQVUsRUFBRSxHQUFHO0lBQ2YsV0FBVyxFQUFFLEtBQUs7SUFDbEIsVUFBVSxFQUFFLEdBQUc7Q0FDaEIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtBZGQsIEtlcm5lbENvbmZpZ30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtjcmVhdGVTaW1wbGVCaW5hcnlLZXJuZWxJbXBsfSBmcm9tICcuLi91dGlscy9iaW5hcnlfaW1wbCc7XG5pbXBvcnQge2JpbmFyeUtlcm5lbEZ1bmMsIGNyZWF0ZUNvbXBsZXhCaW5hcnlLZXJuZWxJbXBsfSBmcm9tICcuLi91dGlscy9iaW5hcnlfdXRpbHMnO1xuXG5leHBvcnQgY29uc3QgYWRkSW1wbCA9XG4gICAgY3JlYXRlU2ltcGxlQmluYXJ5S2VybmVsSW1wbCgoKGE6IG51bWJlciwgYjogbnVtYmVyKSA9PiBhICsgYikpO1xuZXhwb3J0IGNvbnN0IGFkZENvbXBsZXhJbXBsID1cbiAgICBjcmVhdGVDb21wbGV4QmluYXJ5S2VybmVsSW1wbCgoKGFSZWFsLCBhSW1hZywgYlJlYWwsIGJJbWFnKSA9PiB7XG4gICAgICByZXR1cm4ge3JlYWw6IGFSZWFsICsgYlJlYWwsIGltYWc6IGFJbWFnICsgYkltYWd9O1xuICAgIH0pKTtcblxuZXhwb3J0IGNvbnN0IGFkZCA9IGJpbmFyeUtlcm5lbEZ1bmMoQWRkLCBhZGRJbXBsLCBhZGRDb21wbGV4SW1wbCk7XG5cbmV4cG9ydCBjb25zdCBhZGRDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogQWRkLFxuICBiYWNrZW5kTmFtZTogJ2NwdScsXG4gIGtlcm5lbEZ1bmM6IGFkZFxufTtcbiJdfQ==