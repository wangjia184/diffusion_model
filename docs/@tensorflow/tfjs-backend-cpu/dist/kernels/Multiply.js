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
import { Multiply } from '@tensorflow/tfjs-core';
import { createSimpleBinaryKernelImpl } from '../utils/binary_impl';
import { binaryKernelFunc, createComplexBinaryKernelImpl } from '../utils/binary_utils';
export const multiplyImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => aValue * bValue));
export const multiplyComplexImpl = createComplexBinaryKernelImpl(((aReal, aImag, bReal, bImag) => {
    return {
        real: aReal * bReal - aImag * bImag,
        imag: aReal * bImag + aImag * bReal
    };
}));
export const multiply = binaryKernelFunc(Multiply, multiplyImpl, multiplyComplexImpl);
export const multiplyConfig = {
    kernelName: Multiply,
    backendName: 'cpu',
    kernelFunc: multiply
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTXVsdGlwbHkuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtY3B1L3NyYy9rZXJuZWxzL011bHRpcGx5LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBZSxRQUFRLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUM3RCxPQUFPLEVBQUMsNEJBQTRCLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUNsRSxPQUFPLEVBQUMsZ0JBQWdCLEVBQUUsNkJBQTZCLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RixNQUFNLENBQUMsTUFBTSxZQUFZLEdBQUcsNEJBQTRCLENBQ3BELENBQUMsQ0FBQyxNQUFjLEVBQUUsTUFBYyxFQUFFLEVBQUUsQ0FBQyxNQUFNLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQztBQUMzRCxNQUFNLENBQUMsTUFBTSxtQkFBbUIsR0FDNUIsNkJBQTZCLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFFLEtBQUssRUFBRSxFQUFFO0lBQzVELE9BQU87UUFDTCxJQUFJLEVBQUUsS0FBSyxHQUFHLEtBQUssR0FBRyxLQUFLLEdBQUcsS0FBSztRQUNuQyxJQUFJLEVBQUUsS0FBSyxHQUFHLEtBQUssR0FBRyxLQUFLLEdBQUcsS0FBSztLQUNwQyxDQUFDO0FBQ0osQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUVSLE1BQU0sQ0FBQyxNQUFNLFFBQVEsR0FDakIsZ0JBQWdCLENBQUMsUUFBUSxFQUFFLFlBQVksRUFBRSxtQkFBbUIsQ0FBQyxDQUFDO0FBRWxFLE1BQU0sQ0FBQyxNQUFNLGNBQWMsR0FBaUI7SUFDMUMsVUFBVSxFQUFFLFFBQVE7SUFDcEIsV0FBVyxFQUFFLEtBQUs7SUFDbEIsVUFBVSxFQUFFLFFBQVE7Q0FDckIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtLZXJuZWxDb25maWcsIE11bHRpcGx5fSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtjcmVhdGVTaW1wbGVCaW5hcnlLZXJuZWxJbXBsfSBmcm9tICcuLi91dGlscy9iaW5hcnlfaW1wbCc7XG5pbXBvcnQge2JpbmFyeUtlcm5lbEZ1bmMsIGNyZWF0ZUNvbXBsZXhCaW5hcnlLZXJuZWxJbXBsfSBmcm9tICcuLi91dGlscy9iaW5hcnlfdXRpbHMnO1xuXG5leHBvcnQgY29uc3QgbXVsdGlwbHlJbXBsID0gY3JlYXRlU2ltcGxlQmluYXJ5S2VybmVsSW1wbChcbiAgICAoKGFWYWx1ZTogbnVtYmVyLCBiVmFsdWU6IG51bWJlcikgPT4gYVZhbHVlICogYlZhbHVlKSk7XG5leHBvcnQgY29uc3QgbXVsdGlwbHlDb21wbGV4SW1wbCA9XG4gICAgY3JlYXRlQ29tcGxleEJpbmFyeUtlcm5lbEltcGwoKChhUmVhbCwgYUltYWcsIGJSZWFsLCBiSW1hZykgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgcmVhbDogYVJlYWwgKiBiUmVhbCAtIGFJbWFnICogYkltYWcsXG4gICAgICAgIGltYWc6IGFSZWFsICogYkltYWcgKyBhSW1hZyAqIGJSZWFsXG4gICAgICB9O1xuICAgIH0pKTtcblxuZXhwb3J0IGNvbnN0IG11bHRpcGx5ID1cbiAgICBiaW5hcnlLZXJuZWxGdW5jKE11bHRpcGx5LCBtdWx0aXBseUltcGwsIG11bHRpcGx5Q29tcGxleEltcGwpO1xuXG5leHBvcnQgY29uc3QgbXVsdGlwbHlDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogTXVsdGlwbHksXG4gIGJhY2tlbmROYW1lOiAnY3B1JyxcbiAga2VybmVsRnVuYzogbXVsdGlwbHlcbn07XG4iXX0=