/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '../index';
import { BROWSER_ENVS, describeWithFlags, runWithLock } from '../jasmine_util';
import { arrayBufferToBase64String, base64StringToArrayBuffer } from './io_utils';
import { browserLocalStorage, BrowserLocalStorage, BrowserLocalStorageManager, localStorageRouter, purgeLocalStorageArtifacts } from './local_storage';
describeWithFlags('LocalStorage', BROWSER_ENVS, () => {
    // Test data.
    const modelTopology1 = {
        'class_name': 'Sequential',
        'keras_version': '2.1.4',
        'config': [{
                'class_name': 'Dense',
                'config': {
                    'kernel_initializer': {
                        'class_name': 'VarianceScaling',
                        'config': {
                            'distribution': 'uniform',
                            'scale': 1.0,
                            'seed': null,
                            'mode': 'fan_avg'
                        }
                    },
                    'name': 'dense',
                    'kernel_constraint': null,
                    'bias_regularizer': null,
                    'bias_constraint': null,
                    'dtype': 'float32',
                    'activation': 'linear',
                    'trainable': true,
                    'kernel_regularizer': null,
                    'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                    'units': 1,
                    'batch_input_shape': [null, 3],
                    'use_bias': true,
                    'activity_regularizer': null
                }
            }],
        'backend': 'tensorflow'
    };
    const weightSpecs1 = [
        {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
        },
        {
            name: 'dense/bias',
            shape: [1],
            dtype: 'float32',
        }
    ];
    const weightData1 = new ArrayBuffer(16);
    const trainingConfig1 = {
        loss: 'categorical_crossentropy',
        metrics: ['accuracy'],
        optimizer_config: { class_name: 'SGD', config: { learningRate: 0.1 } }
    };
    const artifacts1 = {
        modelTopology: modelTopology1,
        weightSpecs: weightSpecs1,
        weightData: weightData1,
        format: 'layers-model',
        generatedBy: 'TensorFlow.js v0.0.0',
        convertedBy: '1.13.1',
        signature: null,
        userDefinedMetadata: {},
        modelInitializer: {},
        initializerSignature: null,
        trainingConfig: trainingConfig1,
    };
    const artifactsV0 = {
        modelTopology: modelTopology1,
        weightSpecs: weightSpecs1,
        weightData: weightData1
    };
    function findOverflowingByteSize() {
        const LS = window.localStorage;
        const probeKey = `tfjs_test_probe_values_${new Date().getTime()}_${Math.random()}`;
        const minKilobytes = 200;
        const stepKilobytes = 200;
        const maxKilobytes = 40000;
        for (let kilobytes = minKilobytes; kilobytes < maxKilobytes; kilobytes += stepKilobytes) {
            const bytes = kilobytes * 1024;
            const data = new ArrayBuffer(bytes);
            try {
                const encoded = arrayBufferToBase64String(data);
                LS.setItem(probeKey, encoded);
            }
            catch (err) {
                return bytes;
            }
            LS.removeItem(probeKey);
        }
        throw new Error(`Unable to determined overflowing byte size up to ${maxKilobytes} kB.`);
    }
    beforeEach(() => {
        purgeLocalStorageArtifacts();
    });
    afterEach(() => {
        purgeLocalStorageArtifacts();
    });
    it('Save artifacts succeeds', runWithLock(async () => {
        const testStartDate = new Date();
        const handler = tf.io.getSaveHandlers('localstorage://foo/FooModel')[0];
        const saveResult = await handler.save(artifacts1);
        expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
            .toBeGreaterThanOrEqual(testStartDate.getTime());
        // Note: The following two assertions work only because there is no
        //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
        expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
            .toEqual(JSON.stringify(modelTopology1).length);
        expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
            .toEqual(JSON.stringify(weightSpecs1).length);
        expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(16);
        // Check the content of the saved items in local storage.
        const LS = window.localStorage;
        const info = JSON.parse(LS.getItem('tensorflowjs_models/foo/FooModel/info'));
        expect(Date.parse(info.dateSaved))
            .toEqual(saveResult.modelArtifactsInfo.dateSaved.getTime());
        expect(info.modelTopologyBytes)
            .toEqual(saveResult.modelArtifactsInfo.modelTopologyBytes);
        expect(info.weightSpecsBytes)
            .toEqual(saveResult.modelArtifactsInfo.weightSpecsBytes);
        expect(info.weightDataBytes)
            .toEqual(saveResult.modelArtifactsInfo.weightDataBytes);
        const topologyString = LS.getItem('tensorflowjs_models/foo/FooModel/model_topology');
        expect(JSON.stringify(modelTopology1)).toEqual(topologyString);
        const weightSpecsString = LS.getItem('tensorflowjs_models/foo/FooModel/weight_specs');
        expect(JSON.stringify(weightSpecs1)).toEqual(weightSpecsString);
        const weightDataBase64String = LS.getItem('tensorflowjs_models/foo/FooModel/weight_data');
        expect(base64StringToArrayBuffer(weightDataBase64String))
            .toEqual(weightData1);
    }));
    it('Save-load round trip succeeds', runWithLock(async () => {
        const handler1 = tf.io.getSaveHandlers('localstorage://FooModel')[0];
        await handler1.save(artifacts1);
        const handler2 = tf.io.getLoadHandlers('localstorage://FooModel')[0];
        const loaded = await handler2.load();
        expect(loaded.modelTopology).toEqual(modelTopology1);
        expect(loaded.weightSpecs).toEqual(weightSpecs1);
        expect(loaded.weightData).toEqual(weightData1);
        expect(loaded.format).toEqual('layers-model');
        expect(loaded.generatedBy).toEqual('TensorFlow.js v0.0.0');
        expect(loaded.convertedBy).toEqual('1.13.1');
        expect(loaded.userDefinedMetadata).toEqual({});
        expect(loaded.modelInitializer).toEqual({});
        expect(loaded.initializerSignature).toBeUndefined();
        expect(loaded.trainingConfig).toEqual(trainingConfig1);
    }));
    it('Save-load round trip succeeds: v0 format', runWithLock(async () => {
        const handler1 = tf.io.getSaveHandlers('localstorage://FooModel')[0];
        await handler1.save(artifactsV0);
        const handler2 = tf.io.getLoadHandlers('localstorage://FooModel')[0];
        const loaded = await handler2.load();
        expect(loaded.modelTopology).toEqual(modelTopology1);
        expect(loaded.weightSpecs).toEqual(weightSpecs1);
        expect(loaded.weightData).toEqual(weightData1);
        expect(loaded.format).toBeUndefined();
        expect(loaded.generatedBy).toBeUndefined();
        expect(loaded.convertedBy).toBeUndefined();
        expect(loaded.userDefinedMetadata).toBeUndefined();
        expect(loaded.trainingConfig).toBeUndefined();
        expect(loaded.modelInitializer).toBeUndefined();
        expect(loaded.initializerSignature).toBeUndefined();
        expect(loaded.trainingConfig).toBeUndefined();
    }));
    it('Loading nonexistent model fails.', runWithLock(async () => {
        const handler = tf.io.getSaveHandlers('localstorage://NonexistentModel')[0];
        try {
            await handler.load();
        }
        catch (err) {
            expect(err.message)
                .toEqual('In local storage, there is no model with name ' +
                '\'NonexistentModel\'');
            return; // Success
        }
        fail('Loading nonexistent model succeeded unexpectedly.');
    }));
    it('Loading model with missing topology fails.', runWithLock(async () => {
        const handler1 = tf.io.getSaveHandlers('localstorage://FooModel')[0];
        await handler1.save(artifacts1);
        // Manually remove the topology item from local storage.
        window.localStorage.removeItem('tensorflowjs_models/FooModel/model_topology');
        const handler2 = tf.io.getLoadHandlers('localstorage://FooModel')[0];
        try {
            await handler2.load();
        }
        catch (err) {
            expect(err.message)
                .toEqual('In local storage, the topology of model ' +
                '\'FooModel\' is missing.');
            return; // Success
        }
        fail('Loading of model with missing topology succeeded unexpectedly.');
    }));
    it('Loading model with missing weight specs fails.', runWithLock(async () => {
        const handler1 = tf.io.getSaveHandlers('localstorage://FooModel')[0];
        await handler1.save(artifacts1);
        // Manually remove the weight specs item from local storage.
        window.localStorage.removeItem('tensorflowjs_models/FooModel/weight_specs');
        const handler2 = tf.io.getLoadHandlers('localstorage://FooModel')[0];
        try {
            await handler2.load();
        }
        catch (err) {
            expect(err.message)
                .toEqual('In local storage, the weight specs of model ' +
                '\'FooModel\' are missing.');
            return; // Success
        }
        fail('Loading of model with missing weight specs ' +
            'succeeded unexpectedly.');
    }));
    it('Loading model with missing weight data fails.', runWithLock(async () => {
        const handler1 = tf.io.getSaveHandlers('localstorage://FooModel')[0];
        await handler1.save(artifacts1);
        // Manually remove the weight data item from local storage.
        window.localStorage.removeItem('tensorflowjs_models/FooModel/weight_data');
        const handler2 = tf.io.getLoadHandlers('localstorage://FooModel')[0];
        try {
            await handler2.load();
            fail('Loading of model with missing weight data ' +
                'succeeded unexpectedly.');
        }
        catch (err) {
            expect(err.message)
                .toEqual('In local storage, the binary weight values of model ' +
                '\'FooModel\' are missing.');
        }
    }));
    it('Data size too large leads to error thrown', runWithLock(async () => {
        const overflowByteSize = findOverflowingByteSize();
        const overflowArtifacts = {
            modelTopology: modelTopology1,
            weightSpecs: weightSpecs1,
            weightData: new ArrayBuffer(overflowByteSize),
        };
        const handler1 = tf.io.getSaveHandlers('localstorage://FooModel')[0];
        try {
            await handler1.save(overflowArtifacts);
            fail('Saving of model of overflowing-size weight data succeeded ' +
                'unexpectedly.');
        }
        catch (err) {
            expect(err.message
                .indexOf('Failed to save model \'FooModel\' to local storage'))
                .toEqual(0);
        }
    }));
    it('Null, undefined or empty modelPath throws Error', () => {
        expect(() => browserLocalStorage(null))
            .toThrowError(/local storage, modelPath must not be null, undefined or empty/);
        expect(() => browserLocalStorage(undefined))
            .toThrowError(/local storage, modelPath must not be null, undefined or empty/);
        expect(() => browserLocalStorage(''))
            .toThrowError(/local storage, modelPath must not be null, undefined or empty./);
    });
    it('router', () => {
        expect(localStorageRouter('localstorage://bar') instanceof BrowserLocalStorage)
            .toEqual(true);
        expect(localStorageRouter('indexeddb://bar')).toBeNull();
        expect(localStorageRouter('qux')).toBeNull();
    });
    it('Manager: List models: 0 result', runWithLock(async () => {
        // Before any model is saved, listModels should return empty result.
        const out = await new BrowserLocalStorageManager().listModels();
        expect(out).toEqual({});
    }));
    it('Manager: List models: 1 result', runWithLock(async () => {
        const handler = tf.io.getSaveHandlers('localstorage://baz/QuxModel')[0];
        const saveResult = await handler.save(artifacts1);
        // After successful saving, there should be one model.
        const out = await new BrowserLocalStorageManager().listModels();
        if (Object.keys(out).length !== 1) {
            console.log(JSON.stringify(out, null, 2));
        }
        expect(Object.keys(out).length).toEqual(1);
        expect(out['baz/QuxModel'].modelTopologyType)
            .toEqual(saveResult.modelArtifactsInfo.modelTopologyType);
        expect(out['baz/QuxModel'].modelTopologyBytes)
            .toEqual(saveResult.modelArtifactsInfo.modelTopologyBytes);
        expect(out['baz/QuxModel'].weightSpecsBytes)
            .toEqual(saveResult.modelArtifactsInfo.weightSpecsBytes);
        expect(out['baz/QuxModel'].weightDataBytes)
            .toEqual(saveResult.modelArtifactsInfo.weightDataBytes);
    }));
    it('Manager: List models: 2 results', runWithLock(async () => {
        // First, save a model.
        const handler1 = tf.io.getSaveHandlers('localstorage://QuxModel')[0];
        const saveResult1 = await handler1.save(artifacts1);
        // Then, save the model under another path.
        const handler2 = tf.io.getSaveHandlers('localstorage://repeat/QuxModel')[0];
        const saveResult2 = await handler2.save(artifacts1);
        // After successful saving, there should be two models.
        const out = await new BrowserLocalStorageManager().listModels();
        if (Object.keys(out).length !== 2) {
            console.log(JSON.stringify(out, null, 2));
        }
        expect(Object.keys(out).length).toEqual(2);
        expect(out['QuxModel'].modelTopologyType)
            .toEqual(saveResult1.modelArtifactsInfo.modelTopologyType);
        expect(out['QuxModel'].modelTopologyBytes)
            .toEqual(saveResult1.modelArtifactsInfo.modelTopologyBytes);
        expect(out['QuxModel'].weightSpecsBytes)
            .toEqual(saveResult1.modelArtifactsInfo.weightSpecsBytes);
        expect(out['QuxModel'].weightDataBytes)
            .toEqual(saveResult1.modelArtifactsInfo.weightDataBytes);
        expect(out['repeat/QuxModel'].modelTopologyType)
            .toEqual(saveResult2.modelArtifactsInfo.modelTopologyType);
        expect(out['repeat/QuxModel'].modelTopologyBytes)
            .toEqual(saveResult2.modelArtifactsInfo.modelTopologyBytes);
        expect(out['repeat/QuxModel'].weightSpecsBytes)
            .toEqual(saveResult2.modelArtifactsInfo.weightSpecsBytes);
        expect(out['repeat/QuxModel'].weightDataBytes)
            .toEqual(saveResult2.modelArtifactsInfo.weightDataBytes);
    }));
    it('Manager: Successful deleteModel', runWithLock(async () => {
        // First, save a model.
        const handler1 = tf.io.getSaveHandlers('localstorage://QuxModel')[0];
        await handler1.save(artifacts1);
        // Then, save the model under another path.
        const handler2 = tf.io.getSaveHandlers('localstorage://repeat/QuxModel')[0];
        await handler2.save(artifacts1);
        // After successful saving, delete the first save, and then
        // `listModel` should give only one result.
        const manager = new BrowserLocalStorageManager();
        await manager.removeModel('QuxModel');
        const out = await manager.listModels();
        expect(Object.keys(out)).toEqual(['repeat/QuxModel']);
    }));
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibG9jYWxfc3RvcmFnZV90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9pby9sb2NhbF9zdG9yYWdlX3Rlc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxLQUFLLEVBQUUsTUFBTSxVQUFVLENBQUM7QUFDL0IsT0FBTyxFQUFDLFlBQVksRUFBRSxpQkFBaUIsRUFBRSxXQUFXLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUM3RSxPQUFPLEVBQUMseUJBQXlCLEVBQUUseUJBQXlCLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDaEYsT0FBTyxFQUFDLG1CQUFtQixFQUFFLG1CQUFtQixFQUFFLDBCQUEwQixFQUFFLGtCQUFrQixFQUFFLDBCQUEwQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFckosaUJBQWlCLENBQUMsY0FBYyxFQUFFLFlBQVksRUFBRSxHQUFHLEVBQUU7SUFDbkQsYUFBYTtJQUNiLE1BQU0sY0FBYyxHQUFPO1FBQ3pCLFlBQVksRUFBRSxZQUFZO1FBQzFCLGVBQWUsRUFBRSxPQUFPO1FBQ3hCLFFBQVEsRUFBRSxDQUFDO2dCQUNULFlBQVksRUFBRSxPQUFPO2dCQUNyQixRQUFRLEVBQUU7b0JBQ1Isb0JBQW9CLEVBQUU7d0JBQ3BCLFlBQVksRUFBRSxpQkFBaUI7d0JBQy9CLFFBQVEsRUFBRTs0QkFDUixjQUFjLEVBQUUsU0FBUzs0QkFDekIsT0FBTyxFQUFFLEdBQUc7NEJBQ1osTUFBTSxFQUFFLElBQUk7NEJBQ1osTUFBTSxFQUFFLFNBQVM7eUJBQ2xCO3FCQUNGO29CQUNELE1BQU0sRUFBRSxPQUFPO29CQUNmLG1CQUFtQixFQUFFLElBQUk7b0JBQ3pCLGtCQUFrQixFQUFFLElBQUk7b0JBQ3hCLGlCQUFpQixFQUFFLElBQUk7b0JBQ3ZCLE9BQU8sRUFBRSxTQUFTO29CQUNsQixZQUFZLEVBQUUsUUFBUTtv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UUFDckUsTUFBTSxRQUFRLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDO1FBRWhDLDJDQUEyQztRQUMzQyxNQUFNLFFBQVEsR0FDVixFQUFFLENBQUMsRUFBRSxDQUFDLGVBQWUsQ0FBQyxnQ0FBZ0MsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQy9ELE1BQU0sUUFBUSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUVoQywyREFBMkQ7UUFDM0QsMkNBQTJDO1FBQzNDLE1BQU0sT0FBTyxHQUFHLElBQUksMEJBQTBCLEVBQUUsQ0FBQztRQUNqRCxNQUFNLE9BQU8sQ0FBQyxXQUFXLENBQUMsVUFBVSxDQUFDLENBQUM7UUFDdEMsTUFBTSxHQUFHLEdBQUcsTUFBTSxPQUFPLENBQUMsVUFBVSxFQUFFLENBQUM7UUFDdkMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUM7SUFDeEQsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUNULENBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSAnTGljZW5zZScpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gJ0FTIElTJyBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCAqIGFzIHRmIGZyb20gJy4uL2luZGV4JztcbmltcG9ydCB7QlJPV1NFUl9FTlZTLCBkZXNjcmliZVdpdGhGbGFncywgcnVuV2l0aExvY2t9IGZyb20gJy4uL2phc21pbmVfdXRpbCc7XG5pbXBvcnQge2FycmF5QnVmZmVyVG9CYXNlNjRTdHJpbmcsIGJhc2U2NFN0cmluZ1RvQXJyYXlCdWZmZXJ9IGZyb20gJy4vaW9fdXRpbHMnO1xuaW1wb3J0IHticm93c2VyTG9jYWxTdG9yYWdlLCBCcm93c2VyTG9jYWxTdG9yYWdlLCBCcm93c2VyTG9jYWxTdG9yYWdlTWFuYWdlciwgbG9jYWxTdG9yYWdlUm91dGVyLCBwdXJnZUxvY2FsU3RvcmFnZUFydGlmYWN0c30gZnJvbSAnLi9sb2NhbF9zdG9yYWdlJztcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ0xvY2FsU3RvcmFnZScsIEJST1dTRVJfRU5WUywgKCkgPT4ge1xuICAvLyBUZXN0IGRhdGEuXG4gIGNvbnN0IG1vZGVsVG9wb2xvZ3kxOiB7fSA9IHtcbiAgICAnY2xhc3NfbmFtZSc6ICdTZXF1ZW50aWFsJyxcbiAgICAna2VyYXNfdmVyc2lvbic6ICcyLjEuNCcsXG4gICAgJ2NvbmZpZyc6IFt7XG4gICAgICAnY2xhc3NfbmFtZSc6ICdEZW5zZScsXG4gICAgICAnY29uZmlnJzoge1xuICAgICAgICAna2VybmVsX2luaXRpYWxpemVyJzoge1xuICAgICAgICAgICdjbGFzc19uYW1lJzogJ1ZhcmlhbmNlU2NhbGluZycsXG4gICAgICAgICAgJ2NvbmZpZyc6IHtcbiAgICAgICAgICAgICdkaXN0cmlidXRpb24nOiAndW5pZm9ybScsXG4gICAgICAgICAgICAnc2NhbGUnOiAxLjAsXG4gICAgICAgICAgICAnc2VlZCc6IG51bGwsXG4gICAgICAgICAgICAnbW9kZSc6ICdmYW5fYXZnJ1xuICAgICAgICAgIH1cbiAgICAgICAgfSxcbiAgICAgICAgJ25hbWUnOiAnZGVuc2UnLFxuICAgICAgICAna2VybmVsX2NvbnN0cmFpbnQnOiBudWxsLFxuICAgICAgICAnYmlhc19yZWd1bGFyaXplcic6IG51bGwsXG4gICAgICAgICdiaWFzX2NvbnN0cmFpbnQnOiBudWxsLFxuICAgICAgICAnZHR5cGUnOiAnZmxvYXQzMicsXG4gICAgICAgICdhY3RpdmF0aW9uJzogJ2xpbmVhcicsXG4gICAgICAgICd0cmFpbmFibGUnOiB0cnVlLFxuICAgICAgICAna2VybmVsX3JlZ3VsYXJpemVyJzogbnVsbCxcbiAgICAgICAgJ2JpYXNfaW5pdGlhbGl6ZXInOiB7J2NsYXNzX25hbWUnOiAnWmVyb3MnLCAnY29uZmlnJzoge319LFxuICAgICAgICAndW5pdHMnOiAxLFxuICAgICAgICAnYmF0Y2hfaW5wdXRfc2hhcGUnOiBbbnVsbCwgM10sXG4gICAgICAgICd1c2VfYmlhcyc6IHRydWUsXG4gICAgICAgICdhY3Rpdml0eV9yZWd1bGFyaXplcic6IG51bGxcbiAgICAgIH1cbiAgICB9XSxcbiAgICAnYmFja2VuZCc6ICd0ZW5zb3JmbG93J1xuICB9O1xuICBjb25zdCB3ZWlnaHRTcGVjczE6IHRmLmlvLldlaWdodHNNYW5pZmVzdEVudHJ5W10gPSBbXG4gICAge1xuICAgICAgbmFtZTogJ2RlbnNlL2tlcm5lbCcsXG4gICAgICBzaGFwZTogWzMsIDFdLFxuICAgICAgZHR5cGU6ICdmbG9hdDMyJyxcbiAgICB9LFxuICAgIHtcbiAgICAgIG5hbWU6ICdkZW5zZS9iaWFzJyxcbiAgICAgIHNoYXBlOiBbMV0sXG4gICAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICAgIH1cbiAgXTtcbiAgY29uc3Qgd2VpZ2h0RGF0YTEgPSBuZXcgQXJyYXlCdWZmZXIoMTYpO1xuICBjb25zdCB0cmFpbmluZ0NvbmZpZzE6IHRmLmlvLlRyYWluaW5nQ29uZmlnID0ge1xuICAgIGxvc3M6ICdjYXRlZ29yaWNhbF9jcm9zc2VudHJvcHknLFxuICAgIG1ldHJpY3M6IFsnYWNjdXJhY3knXSxcbiAgICBvcHRpbWl6ZXJfY29uZmlnOiB7Y2xhc3NfbmFtZTogJ1NHRCcsIGNvbmZpZzoge2xlYXJuaW5nUmF0ZTogMC4xfX1cbiAgfTtcblxuICBjb25zdCBhcnRpZmFjdHMxOiB0Zi5pby5Nb2RlbEFydGlmYWN0cyA9IHtcbiAgICBtb2RlbFRvcG9sb2d5OiBtb2RlbFRvcG9sb2d5MSxcbiAgICB3ZWlnaHRTcGVjczogd2VpZ2h0U3BlY3MxLFxuICAgIHdlaWdodERhdGE6IHdlaWdodERhdGExLFxuICAgIGZvcm1hdDogJ2xheWVycy1tb2RlbCcsXG4gICAgZ2VuZXJhdGVkQnk6ICdUZW5zb3JGbG93LmpzIHYwLjAuMCcsXG4gICAgY29udmVydGVkQnk6ICcxLjEzLjEnLFxuICAgIHNpZ25hdHVyZTogbnVsbCxcbiAgICB1c2VyRGVmaW5lZE1ldGFkYXRhOiB7fSxcbiAgICBtb2RlbEluaXRpYWxpemVyOiB7fSxcbiAgICBpbml0aWFsaXplclNpZ25hdHVyZTogbnVsbCxcbiAgICB0cmFpbmluZ0NvbmZpZzogdHJhaW5pbmdDb25maWcxLFxuICB9O1xuXG4gIGNvbnN0IGFydGlmYWN0c1YwOiB0Zi5pby5Nb2RlbEFydGlmYWN0cyA9IHtcbiAgICBtb2RlbFRvcG9sb2d5OiBtb2RlbFRvcG9sb2d5MSxcbiAgICB3ZWlnaHRTcGVjczogd2VpZ2h0U3BlY3MxLFxuICAgIHdlaWdodERhdGE6IHdlaWdodERhdGExXG4gIH07XG5cbiAgZnVuY3Rpb24gZmluZE92ZXJmbG93aW5nQnl0ZVNpemUoKTogbnVtYmVyIHtcbiAgICBjb25zdCBMUyA9IHdpbmRvdy5sb2NhbFN0b3JhZ2U7XG4gICAgY29uc3QgcHJvYmVLZXkgPVxuICAgICAgICBgdGZqc190ZXN0X3Byb2JlX3ZhbHVlc18ke25ldyBEYXRlKCkuZ2V0VGltZSgpfV8ke01hdGgucmFuZG9tKCl9YDtcbiAgICBjb25zdCBtaW5LaWxvYnl0ZXMgPSAyMDA7XG4gICAgY29uc3Qgc3RlcEtpbG9ieXRlcyA9IDIwMDtcbiAgICBjb25zdCBtYXhLaWxvYnl0ZXMgPSA0MDAwMDtcbiAgICBmb3IgKGxldCBraWxvYnl0ZXMgPSBtaW5LaWxvYnl0ZXM7IGtpbG9ieXRlcyA8IG1heEtpbG9ieXRlcztcbiAgICAgICAgIGtpbG9ieXRlcyArPSBzdGVwS2lsb2J5dGVzKSB7XG4gICAgICBjb25zdCBieXRlcyA9IGtpbG9ieXRlcyAqIDEwMjQ7XG4gICAgICBjb25zdCBkYXRhID0gbmV3IEFycmF5QnVmZmVyKGJ5dGVzKTtcbiAgICAgIHRyeSB7XG4gICAgICAgIGNvbnN0IGVuY29kZWQgPSBhcnJheUJ1ZmZlclRvQmFzZTY0U3RyaW5nKGRhdGEpO1xuICAgICAgICBMUy5zZXRJdGVtKHByb2JlS2V5LCBlbmNvZGVkKTtcbiAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICByZXR1cm4gYnl0ZXM7XG4gICAgICB9XG4gICAgICBMUy5yZW1vdmVJdGVtKHByb2JlS2V5KTtcbiAgICB9XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgVW5hYmxlIHRvIGRldGVybWluZWQgb3ZlcmZsb3dpbmcgYnl0ZSBzaXplIHVwIHRvICR7bWF4S2lsb2J5dGVzfSBrQi5gKTtcbiAgfVxuXG4gIGJlZm9yZUVhY2goKCkgPT4ge1xuICAgIHB1cmdlTG9jYWxTdG9yYWdlQXJ0aWZhY3RzKCk7XG4gIH0pO1xuXG4gIGFmdGVyRWFjaCgoKSA9PiB7XG4gICAgcHVyZ2VMb2NhbFN0b3JhZ2VBcnRpZmFjdHMoKTtcbiAgfSk7XG5cbiAgaXQoJ1NhdmUgYXJ0aWZhY3RzIHN1Y2NlZWRzJywgcnVuV2l0aExvY2soYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IHRlc3RTdGFydERhdGUgPSBuZXcgRGF0ZSgpO1xuICAgICAgIGNvbnN0IGhhbmRsZXIgPSB0Zi5pby5nZXRTYXZlSGFuZGxlcnMoJ2xvY2Fsc3RvcmFnZTovL2Zvby9Gb29Nb2RlbCcpWzBdO1xuICAgICAgIGNvbnN0IHNhdmVSZXN1bHQgPSBhd2FpdCBoYW5kbGVyLnNhdmUoYXJ0aWZhY3RzMSk7XG5cbiAgICAgICBleHBlY3Qoc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm8uZGF0ZVNhdmVkLmdldFRpbWUoKSlcbiAgICAgICAgICAgLnRvQmVHcmVhdGVyVGhhbk9yRXF1YWwodGVzdFN0YXJ0RGF0ZS5nZXRUaW1lKCkpO1xuICAgICAgIC8vIE5vdGU6IFRoZSBmb2xsb3dpbmcgdHdvIGFzc2VydGlvbnMgd29yayBvbmx5IGJlY2F1c2UgdGhlcmUgaXMgbm9cbiAgICAgICAvLyAgIG5vbi1BU0NJSSBjaGFyYWN0ZXJzIGluIGBtb2RlbFRvcG9sb2d5MWAgYW5kIGB3ZWlnaHRTcGVjczFgLlxuICAgICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby5tb2RlbFRvcG9sb2d5Qnl0ZXMpXG4gICAgICAgICAgIC50b0VxdWFsKEpTT04uc3RyaW5naWZ5KG1vZGVsVG9wb2xvZ3kxKS5sZW5ndGgpO1xuICAgICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHRTcGVjc0J5dGVzKVxuICAgICAgICAgICAudG9FcXVhbChKU09OLnN0cmluZ2lmeSh3ZWlnaHRTcGVjczEpLmxlbmd0aCk7XG4gICAgICAgZXhwZWN0KHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvLndlaWdodERhdGFCeXRlcykudG9FcXVhbCgxNik7XG5cbiAgICAgICAvLyBDaGVjayB0aGUgY29udGVudCBvZiB0aGUgc2F2ZWQgaXRlbXMgaW4gbG9jYWwgc3RvcmFnZS5cbiAgICAgICBjb25zdCBMUyA9IHdpbmRvdy5sb2NhbFN0b3JhZ2U7XG4gICAgICAgY29uc3QgaW5mbyA9XG4gICAgICAgICAgIEpTT04ucGFyc2UoTFMuZ2V0SXRlbSgndGVuc29yZmxvd2pzX21vZGVscy9mb28vRm9vTW9kZWwvaW5mbycpKTtcbiAgICAgICBleHBlY3QoRGF0ZS5wYXJzZShpbmZvLmRhdGVTYXZlZCkpXG4gICAgICAgICAgIC50b0VxdWFsKHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvLmRhdGVTYXZlZC5nZXRUaW1lKCkpO1xuICAgICAgIGV4cGVjdChpbmZvLm1vZGVsVG9wb2xvZ3lCeXRlcylcbiAgICAgICAgICAgLnRvRXF1YWwoc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm8ubW9kZWxUb3BvbG9neUJ5dGVzKTtcbiAgICAgICBleHBlY3QoaW5mby53ZWlnaHRTcGVjc0J5dGVzKVxuICAgICAgICAgICAudG9FcXVhbChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHRTcGVjc0J5dGVzKTtcbiAgICAgICBleHBlY3QoaW5mby53ZWlnaHREYXRhQnl0ZXMpXG4gICAgICAgICAgIC50b0VxdWFsKHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvLndlaWdodERhdGFCeXRlcyk7XG5cbiAgICAgICBjb25zdCB0b3BvbG9neVN0cmluZyA9XG4gICAgICAgICAgIExTLmdldEl0ZW0oJ3RlbnNvcmZsb3dqc19tb2RlbHMvZm9vL0Zvb01vZGVsL21vZGVsX3RvcG9sb2d5Jyk7XG4gICAgICAgZXhwZWN0KEpTT04uc3RyaW5naWZ5KG1vZGVsVG9wb2xvZ3kxKSkudG9FcXVhbCh0b3BvbG9neVN0cmluZyk7XG5cbiAgICAgICBjb25zdCB3ZWlnaHRTcGVjc1N0cmluZyA9XG4gICAgICAgICAgIExTLmdldEl0ZW0oJ3RlbnNvcmZsb3dqc19tb2RlbHMvZm9vL0Zvb01vZGVsL3dlaWdodF9zcGVjcycpO1xuICAgICAgIGV4cGVjdChKU09OLnN0cmluZ2lmeSh3ZWlnaHRTcGVjczEpKS50b0VxdWFsKHdlaWdodFNwZWNzU3RyaW5nKTtcblxuICAgICAgIGNvbnN0IHdlaWdodERhdGFCYXNlNjRTdHJpbmcgPVxuICAgICAgICAgICBMUy5nZXRJdGVtKCd0ZW5zb3JmbG93anNfbW9kZWxzL2Zvby9Gb29Nb2RlbC93ZWlnaHRfZGF0YScpO1xuICAgICAgIGV4cGVjdChiYXNlNjRTdHJpbmdUb0FycmF5QnVmZmVyKHdlaWdodERhdGFCYXNlNjRTdHJpbmcpKVxuICAgICAgICAgICAudG9FcXVhbCh3ZWlnaHREYXRhMSk7XG4gICAgIH0pKTtcblxuICBpdCgnU2F2ZS1sb2FkIHJvdW5kIHRyaXAgc3VjY2VlZHMnLCBydW5XaXRoTG9jayhhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaGFuZGxlcjEgPSB0Zi5pby5nZXRTYXZlSGFuZGxlcnMoJ2xvY2Fsc3RvcmFnZTovL0Zvb01vZGVsJylbMF07XG5cbiAgICAgICBhd2FpdCBoYW5kbGVyMS5zYXZlKGFydGlmYWN0czEpO1xuICAgICAgIGNvbnN0IGhhbmRsZXIyID0gdGYuaW8uZ2V0TG9hZEhhbmRsZXJzKCdsb2NhbHN0b3JhZ2U6Ly9Gb29Nb2RlbCcpWzBdO1xuICAgICAgIGNvbnN0IGxvYWRlZCA9IGF3YWl0IGhhbmRsZXIyLmxvYWQoKTtcbiAgICAgICBleHBlY3QobG9hZGVkLm1vZGVsVG9wb2xvZ3kpLnRvRXF1YWwobW9kZWxUb3BvbG9neTEpO1xuICAgICAgIGV4cGVjdChsb2FkZWQud2VpZ2h0U3BlY3MpLnRvRXF1YWwod2VpZ2h0U3BlY3MxKTtcbiAgICAgICBleHBlY3QobG9hZGVkLndlaWdodERhdGEpLnRvRXF1YWwod2VpZ2h0RGF0YTEpO1xuICAgICAgIGV4cGVjdChsb2FkZWQuZm9ybWF0KS50b0VxdWFsKCdsYXllcnMtbW9kZWwnKTtcbiAgICAgICBleHBlY3QobG9hZGVkLmdlbmVyYXRlZEJ5KS50b0VxdWFsKCdUZW5zb3JGbG93LmpzIHYwLjAuMCcpO1xuICAgICAgIGV4cGVjdChsb2FkZWQuY29udmVydGVkQnkpLnRvRXF1YWwoJzEuMTMuMScpO1xuICAgICAgIGV4cGVjdChsb2FkZWQudXNlckRlZmluZWRNZXRhZGF0YSkudG9FcXVhbCh7fSk7XG4gICAgICAgZXhwZWN0KGxvYWRlZC5tb2RlbEluaXRpYWxpemVyKS50b0VxdWFsKHt9KTtcbiAgICAgICBleHBlY3QobG9hZGVkLmluaXRpYWxpemVyU2lnbmF0dXJlKS50b0JlVW5kZWZpbmVkKCk7XG4gICAgICAgZXhwZWN0KGxvYWRlZC50cmFpbmluZ0NvbmZpZykudG9FcXVhbCh0cmFpbmluZ0NvbmZpZzEpO1xuICAgICB9KSk7XG5cbiAgaXQoJ1NhdmUtbG9hZCByb3VuZCB0cmlwIHN1Y2NlZWRzOiB2MCBmb3JtYXQnLCBydW5XaXRoTG9jayhhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaGFuZGxlcjEgPSB0Zi5pby5nZXRTYXZlSGFuZGxlcnMoJ2xvY2Fsc3RvcmFnZTovL0Zvb01vZGVsJylbMF07XG5cbiAgICAgICBhd2FpdCBoYW5kbGVyMS5zYXZlKGFydGlmYWN0c1YwKTtcbiAgICAgICBjb25zdCBoYW5kbGVyMiA9IHRmLmlvLmdldExvYWRIYW5kbGVycygnbG9jYWxzdG9yYWdlOi8vRm9vTW9kZWwnKVswXTtcbiAgICAgICBjb25zdCBsb2FkZWQgPSBhd2FpdCBoYW5kbGVyMi5sb2FkKCk7XG4gICAgICAgZXhwZWN0KGxvYWRlZC5tb2RlbFRvcG9sb2d5KS50b0VxdWFsKG1vZGVsVG9wb2xvZ3kxKTtcbiAgICAgICBleHBlY3QobG9hZGVkLndlaWdodFNwZWNzKS50b0VxdWFsKHdlaWdodFNwZWNzMSk7XG4gICAgICAgZXhwZWN0KGxvYWRlZC53ZWlnaHREYXRhKS50b0VxdWFsKHdlaWdodERhdGExKTtcbiAgICAgICBleHBlY3QobG9hZGVkLmZvcm1hdCkudG9CZVVuZGVmaW5lZCgpO1xuICAgICAgIGV4cGVjdChsb2FkZWQuZ2VuZXJhdGVkQnkpLnRvQmVVbmRlZmluZWQoKTtcbiAgICAgICBleHBlY3QobG9hZGVkLmNvbnZlcnRlZEJ5KS50b0JlVW5kZWZpbmVkKCk7XG4gICAgICAgZXhwZWN0KGxvYWRlZC51c2VyRGVmaW5lZE1ldGFkYXRhKS50b0JlVW5kZWZpbmVkKCk7XG4gICAgICAgZXhwZWN0KGxvYWRlZC50cmFpbmluZ0NvbmZpZykudG9CZVVuZGVmaW5lZCgpO1xuICAgICAgIGV4cGVjdChsb2FkZWQubW9kZWxJbml0aWFsaXplcikudG9CZVVuZGVmaW5lZCgpO1xuICAgICAgIGV4cGVjdChsb2FkZWQuaW5pdGlhbGl6ZXJTaWduYXR1cmUpLnRvQmVVbmRlZmluZWQoKTtcbiAgICAgICBleHBlY3QobG9hZGVkLnRyYWluaW5nQ29uZmlnKS50b0JlVW5kZWZpbmVkKCk7XG4gICAgIH0pKTtcblxuICBpdCgnTG9hZGluZyBub25leGlzdGVudCBtb2RlbCBmYWlscy4nLCBydW5XaXRoTG9jayhhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaGFuZGxlciA9XG4gICAgICAgICAgIHRmLmlvLmdldFNhdmVIYW5kbGVycygnbG9jYWxzdG9yYWdlOi8vTm9uZXhpc3RlbnRNb2RlbCcpWzBdO1xuICAgICAgIHRyeSB7XG4gICAgICAgICBhd2FpdCBoYW5kbGVyLmxvYWQoKTtcbiAgICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgIGV4cGVjdChlcnIubWVzc2FnZSlcbiAgICAgICAgICAgICAudG9FcXVhbChcbiAgICAgICAgICAgICAgICAgJ0luIGxvY2FsIHN0b3JhZ2UsIHRoZXJlIGlzIG5vIG1vZGVsIHdpdGggbmFtZSAnICtcbiAgICAgICAgICAgICAgICAgJ1xcJ05vbmV4aXN0ZW50TW9kZWxcXCcnKTtcbiAgICAgICAgIHJldHVybjsgIC8vIFN1Y2Nlc3NcbiAgICAgICB9XG4gICAgICAgZmFpbCgnTG9hZGluZyBub25leGlzdGVudCBtb2RlbCBzdWNjZWVkZWQgdW5leHBlY3RlZGx5LicpO1xuICAgICB9KSk7XG5cbiAgaXQoJ0xvYWRpbmcgbW9kZWwgd2l0aCBtaXNzaW5nIHRvcG9sb2d5IGZhaWxzLicsIHJ1bldpdGhMb2NrKGFzeW5jICgpID0+IHtcbiAgICAgICBjb25zdCBoYW5kbGVyMSA9IHRmLmlvLmdldFNhdmVIYW5kbGVycygnbG9jYWxzdG9yYWdlOi8vRm9vTW9kZWwnKVswXTtcbiAgICAgICBhd2FpdCBoYW5kbGVyMS5zYXZlKGFydGlmYWN0czEpO1xuICAgICAgIC8vIE1hbnVhbGx5IHJlbW92ZSB0aGUgdG9wb2xvZ3kgaXRlbSBmcm9tIGxvY2FsIHN0b3JhZ2UuXG4gICAgICAgd2luZG93LmxvY2FsU3RvcmFnZS5yZW1vdmVJdGVtKFxuICAgICAgICAgICAndGVuc29yZmxvd2pzX21vZGVscy9Gb29Nb2RlbC9tb2RlbF90b3BvbG9neScpO1xuXG4gICAgICAgY29uc3QgaGFuZGxlcjIgPSB0Zi5pby5nZXRMb2FkSGFuZGxlcnMoJ2xvY2Fsc3RvcmFnZTovL0Zvb01vZGVsJylbMF07XG4gICAgICAgdHJ5IHtcbiAgICAgICAgIGF3YWl0IGhhbmRsZXIyLmxvYWQoKTtcbiAgICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgIGV4cGVjdChlcnIubWVzc2FnZSlcbiAgICAgICAgICAgICAudG9FcXVhbChcbiAgICAgICAgICAgICAgICAgJ0luIGxvY2FsIHN0b3JhZ2UsIHRoZSB0b3BvbG9neSBvZiBtb2RlbCAnICtcbiAgICAgICAgICAgICAgICAgJ1xcJ0Zvb01vZGVsXFwnIGlzIG1pc3NpbmcuJyk7XG4gICAgICAgICByZXR1cm47ICAvLyBTdWNjZXNzXG4gICAgICAgfVxuICAgICAgIGZhaWwoJ0xvYWRpbmcgb2YgbW9kZWwgd2l0aCBtaXNzaW5nIHRvcG9sb2d5IHN1Y2NlZWRlZCB1bmV4cGVjdGVkbHkuJyk7XG4gICAgIH0pKTtcblxuICBpdCgnTG9hZGluZyBtb2RlbCB3aXRoIG1pc3Npbmcgd2VpZ2h0IHNwZWNzIGZhaWxzLicsIHJ1bldpdGhMb2NrKGFzeW5jICgpID0+IHtcbiAgICAgICBjb25zdCBoYW5kbGVyMSA9IHRmLmlvLmdldFNhdmVIYW5kbGVycygnbG9jYWxzdG9yYWdlOi8vRm9vTW9kZWwnKVswXTtcbiAgICAgICBhd2FpdCBoYW5kbGVyMS5zYXZlKGFydGlmYWN0czEpO1xuICAgICAgIC8vIE1hbnVhbGx5IHJlbW92ZSB0aGUgd2VpZ2h0IHNwZWNzIGl0ZW0gZnJvbSBsb2NhbCBzdG9yYWdlLlxuICAgICAgIHdpbmRvdy5sb2NhbFN0b3JhZ2UucmVtb3ZlSXRlbShcbiAgICAgICAgICAgJ3RlbnNvcmZsb3dqc19tb2RlbHMvRm9vTW9kZWwvd2VpZ2h0X3NwZWNzJyk7XG5cbiAgICAgICBjb25zdCBoYW5kbGVyMiA9IHRmLmlvLmdldExvYWRIYW5kbGVycygnbG9jYWxzdG9yYWdlOi8vRm9vTW9kZWwnKVswXTtcbiAgICAgICB0cnkge1xuICAgICAgICAgYXdhaXQgaGFuZGxlcjIubG9hZCgpO1xuICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgZXhwZWN0KGVyci5tZXNzYWdlKVxuICAgICAgICAgICAgIC50b0VxdWFsKFxuICAgICAgICAgICAgICAgICAnSW4gbG9jYWwgc3RvcmFnZSwgdGhlIHdlaWdodCBzcGVjcyBvZiBtb2RlbCAnICtcbiAgICAgICAgICAgICAgICAgJ1xcJ0Zvb01vZGVsXFwnIGFyZSBtaXNzaW5nLicpO1xuICAgICAgICAgcmV0dXJuOyAgLy8gU3VjY2Vzc1xuICAgICAgIH1cbiAgICAgICBmYWlsKFxuICAgICAgICAgICAnTG9hZGluZyBvZiBtb2RlbCB3aXRoIG1pc3Npbmcgd2VpZ2h0IHNwZWNzICcgK1xuICAgICAgICAgICAnc3VjY2VlZGVkIHVuZXhwZWN0ZWRseS4nKTtcbiAgICAgfSkpO1xuXG4gIGl0KCdMb2FkaW5nIG1vZGVsIHdpdGggbWlzc2luZyB3ZWlnaHQgZGF0YSBmYWlscy4nLCBydW5XaXRoTG9jayhhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaGFuZGxlcjEgPSB0Zi5pby5nZXRTYXZlSGFuZGxlcnMoJ2xvY2Fsc3RvcmFnZTovL0Zvb01vZGVsJylbMF07XG4gICAgICAgYXdhaXQgaGFuZGxlcjEuc2F2ZShhcnRpZmFjdHMxKTtcblxuICAgICAgIC8vIE1hbnVhbGx5IHJlbW92ZSB0aGUgd2VpZ2h0IGRhdGEgaXRlbSBmcm9tIGxvY2FsIHN0b3JhZ2UuXG4gICAgICAgd2luZG93LmxvY2FsU3RvcmFnZS5yZW1vdmVJdGVtKFxuICAgICAgICAgICAndGVuc29yZmxvd2pzX21vZGVscy9Gb29Nb2RlbC93ZWlnaHRfZGF0YScpO1xuXG4gICAgICAgY29uc3QgaGFuZGxlcjIgPSB0Zi5pby5nZXRMb2FkSGFuZGxlcnMoJ2xvY2Fsc3RvcmFnZTovL0Zvb01vZGVsJylbMF07XG4gICAgICAgdHJ5IHtcbiAgICAgICAgIGF3YWl0IGhhbmRsZXIyLmxvYWQoKTtcbiAgICAgICAgIGZhaWwoXG4gICAgICAgICAgICAgJ0xvYWRpbmcgb2YgbW9kZWwgd2l0aCBtaXNzaW5nIHdlaWdodCBkYXRhICcgK1xuICAgICAgICAgICAgICdzdWNjZWVkZWQgdW5leHBlY3RlZGx5LicpO1xuICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgZXhwZWN0KGVyci5tZXNzYWdlKVxuICAgICAgICAgICAgIC50b0VxdWFsKFxuICAgICAgICAgICAgICAgICAnSW4gbG9jYWwgc3RvcmFnZSwgdGhlIGJpbmFyeSB3ZWlnaHQgdmFsdWVzIG9mIG1vZGVsICcgK1xuICAgICAgICAgICAgICAgICAnXFwnRm9vTW9kZWxcXCcgYXJlIG1pc3NpbmcuJyk7XG4gICAgICAgfVxuICAgICB9KSk7XG5cbiAgaXQoJ0RhdGEgc2l6ZSB0b28gbGFyZ2UgbGVhZHMgdG8gZXJyb3IgdGhyb3duJywgcnVuV2l0aExvY2soYXN5bmMgKCkgPT4ge1xuICAgICAgIGNvbnN0IG92ZXJmbG93Qnl0ZVNpemUgPSBmaW5kT3ZlcmZsb3dpbmdCeXRlU2l6ZSgpO1xuICAgICAgIGNvbnN0IG92ZXJmbG93QXJ0aWZhY3RzOiB0Zi5pby5Nb2RlbEFydGlmYWN0cyA9IHtcbiAgICAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxLFxuICAgICAgICAgd2VpZ2h0U3BlY3M6IHdlaWdodFNwZWNzMSxcbiAgICAgICAgIHdlaWdodERhdGE6IG5ldyBBcnJheUJ1ZmZlcihvdmVyZmxvd0J5dGVTaXplKSxcbiAgICAgICB9O1xuICAgICAgIGNvbnN0IGhhbmRsZXIxID0gdGYuaW8uZ2V0U2F2ZUhhbmRsZXJzKCdsb2NhbHN0b3JhZ2U6Ly9Gb29Nb2RlbCcpWzBdO1xuICAgICAgIHRyeSB7XG4gICAgICAgICBhd2FpdCBoYW5kbGVyMS5zYXZlKG92ZXJmbG93QXJ0aWZhY3RzKTtcbiAgICAgICAgIGZhaWwoXG4gICAgICAgICAgICAgJ1NhdmluZyBvZiBtb2RlbCBvZiBvdmVyZmxvd2luZy1zaXplIHdlaWdodCBkYXRhIHN1Y2NlZWRlZCAnICtcbiAgICAgICAgICAgICAndW5leHBlY3RlZGx5LicpO1xuICAgICAgIH0gY2F0Y2ggKGVycikge1xuICAgICAgICAgZXhwZWN0KFxuICAgICAgICAgICAgIChlcnIubWVzc2FnZSBhcyBzdHJpbmcpXG4gICAgICAgICAgICAgICAgIC5pbmRleE9mKCdGYWlsZWQgdG8gc2F2ZSBtb2RlbCBcXCdGb29Nb2RlbFxcJyB0byBsb2NhbCBzdG9yYWdlJykpXG4gICAgICAgICAgICAgLnRvRXF1YWwoMCk7XG4gICAgICAgfVxuICAgICB9KSk7XG5cbiAgaXQoJ051bGwsIHVuZGVmaW5lZCBvciBlbXB0eSBtb2RlbFBhdGggdGhyb3dzIEVycm9yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiBicm93c2VyTG9jYWxTdG9yYWdlKG51bGwpKVxuICAgICAgICAudG9UaHJvd0Vycm9yKFxuICAgICAgICAgICAgL2xvY2FsIHN0b3JhZ2UsIG1vZGVsUGF0aCBtdXN0IG5vdCBiZSBudWxsLCB1bmRlZmluZWQgb3IgZW1wdHkvKTtcbiAgICBleHBlY3QoKCkgPT4gYnJvd3NlckxvY2FsU3RvcmFnZSh1bmRlZmluZWQpKVxuICAgICAgICAudG9UaHJvd0Vycm9yKFxuICAgICAgICAgICAgL2xvY2FsIHN0b3JhZ2UsIG1vZGVsUGF0aCBtdXN0IG5vdCBiZSBudWxsLCB1bmRlZmluZWQgb3IgZW1wdHkvKTtcbiAgICBleHBlY3QoKCkgPT4gYnJvd3NlckxvY2FsU3RvcmFnZSgnJykpXG4gICAgICAgIC50b1Rocm93RXJyb3IoXG4gICAgICAgICAgICAvbG9jYWwgc3RvcmFnZSwgbW9kZWxQYXRoIG11c3Qgbm90IGJlIG51bGwsIHVuZGVmaW5lZCBvciBlbXB0eS4vKTtcbiAgfSk7XG5cbiAgaXQoJ3JvdXRlcicsICgpID0+IHtcbiAgICBleHBlY3QoXG4gICAgICAgIGxvY2FsU3RvcmFnZVJvdXRlcignbG9jYWxzdG9yYWdlOi8vYmFyJykgaW5zdGFuY2VvZiBCcm93c2VyTG9jYWxTdG9yYWdlKVxuICAgICAgICAudG9FcXVhbCh0cnVlKTtcbiAgICBleHBlY3QobG9jYWxTdG9yYWdlUm91dGVyKCdpbmRleGVkZGI6Ly9iYXInKSkudG9CZU51bGwoKTtcbiAgICBleHBlY3QobG9jYWxTdG9yYWdlUm91dGVyKCdxdXgnKSkudG9CZU51bGwoKTtcbiAgfSk7XG5cbiAgaXQoJ01hbmFnZXI6IExpc3QgbW9kZWxzOiAwIHJlc3VsdCcsIHJ1bldpdGhMb2NrKGFzeW5jICgpID0+IHtcbiAgICAgICAvLyBCZWZvcmUgYW55IG1vZGVsIGlzIHNhdmVkLCBsaXN0TW9kZWxzIHNob3VsZCByZXR1cm4gZW1wdHkgcmVzdWx0LlxuICAgICAgIGNvbnN0IG91dCA9IGF3YWl0IG5ldyBCcm93c2VyTG9jYWxTdG9yYWdlTWFuYWdlcigpLmxpc3RNb2RlbHMoKTtcbiAgICAgICBleHBlY3Qob3V0KS50b0VxdWFsKHt9KTtcbiAgICAgfSkpO1xuXG4gIGl0KCdNYW5hZ2VyOiBMaXN0IG1vZGVsczogMSByZXN1bHQnLCBydW5XaXRoTG9jayhhc3luYyAoKSA9PiB7XG4gICAgICAgY29uc3QgaGFuZGxlciA9IHRmLmlvLmdldFNhdmVIYW5kbGVycygnbG9jYWxzdG9yYWdlOi8vYmF6L1F1eE1vZGVsJylbMF07XG4gICAgICAgY29uc3Qgc2F2ZVJlc3VsdCA9IGF3YWl0IGhhbmRsZXIuc2F2ZShhcnRpZmFjdHMxKTtcblxuICAgICAgIC8vIEFmdGVyIHN1Y2Nlc3NmdWwgc2F2aW5nLCB0aGVyZSBzaG91bGQgYmUgb25lIG1vZGVsLlxuICAgICAgIGNvbnN0IG91dCA9IGF3YWl0IG5ldyBCcm93c2VyTG9jYWxTdG9yYWdlTWFuYWdlcigpLmxpc3RNb2RlbHMoKTtcbiAgICAgICBpZiAoT2JqZWN0LmtleXMob3V0KS5sZW5ndGggIT09IDEpIHtcbiAgICAgICAgIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG91dCwgbnVsbCwgMikpO1xuICAgICAgIH1cblxuICAgICAgIGV4cGVjdChPYmplY3Qua2V5cyhvdXQpLmxlbmd0aCkudG9FcXVhbCgxKTtcbiAgICAgICBleHBlY3Qob3V0WydiYXovUXV4TW9kZWwnXS5tb2RlbFRvcG9sb2d5VHlwZSlcbiAgICAgICAgICAgLnRvRXF1YWwoc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm8ubW9kZWxUb3BvbG9neVR5cGUpO1xuICAgICAgIGV4cGVjdChvdXRbJ2Jhei9RdXhNb2RlbCddLm1vZGVsVG9wb2xvZ3lCeXRlcylcbiAgICAgICAgICAgLnRvRXF1YWwoc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm8ubW9kZWxUb3BvbG9neUJ5dGVzKTtcbiAgICAgICBleHBlY3Qob3V0WydiYXovUXV4TW9kZWwnXS53ZWlnaHRTcGVjc0J5dGVzKVxuICAgICAgICAgICAudG9FcXVhbChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHRTcGVjc0J5dGVzKTtcbiAgICAgICBleHBlY3Qob3V0WydiYXovUXV4TW9kZWwnXS53ZWlnaHREYXRhQnl0ZXMpXG4gICAgICAgICAgIC50b0VxdWFsKHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvLndlaWdodERhdGFCeXRlcyk7XG4gICAgIH0pKTtcblxuICBpdCgnTWFuYWdlcjogTGlzdCBtb2RlbHM6IDIgcmVzdWx0cycsIHJ1bldpdGhMb2NrKGFzeW5jICgpID0+IHtcbiAgICAgICAvLyBGaXJzdCwgc2F2ZSBhIG1vZGVsLlxuICAgICAgIGNvbnN0IGhhbmRsZXIxID0gdGYuaW8uZ2V0U2F2ZUhhbmRsZXJzKCdsb2NhbHN0b3JhZ2U6Ly9RdXhNb2RlbCcpWzBdO1xuICAgICAgIGNvbnN0IHNhdmVSZXN1bHQxID0gYXdhaXQgaGFuZGxlcjEuc2F2ZShhcnRpZmFjdHMxKTtcblxuICAgICAgIC8vIFRoZW4sIHNhdmUgdGhlIG1vZGVsIHVuZGVyIGFub3RoZXIgcGF0aC5cbiAgICAgICBjb25zdCBoYW5kbGVyMiA9XG4gICAgICAgICAgIHRmLmlvLmdldFNhdmVIYW5kbGVycygnbG9jYWxzdG9yYWdlOi8vcmVwZWF0L1F1eE1vZGVsJylbMF07XG4gICAgICAgY29uc3Qgc2F2ZVJlc3VsdDIgPSBhd2FpdCBoYW5kbGVyMi5zYXZlKGFydGlmYWN0czEpO1xuXG4gICAgICAgLy8gQWZ0ZXIgc3VjY2Vzc2Z1bCBzYXZpbmcsIHRoZXJlIHNob3VsZCBiZSB0d28gbW9kZWxzLlxuICAgICAgIGNvbnN0IG91dCA9IGF3YWl0IG5ldyBCcm93c2VyTG9jYWxTdG9yYWdlTWFuYWdlcigpLmxpc3RNb2RlbHMoKTtcbiAgICAgICBpZiAoT2JqZWN0LmtleXMob3V0KS5sZW5ndGggIT09IDIpIHtcbiAgICAgICAgIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG91dCwgbnVsbCwgMikpO1xuICAgICAgIH1cbiAgICAgICBleHBlY3QoT2JqZWN0LmtleXMob3V0KS5sZW5ndGgpLnRvRXF1YWwoMik7XG4gICAgICAgZXhwZWN0KG91dFsnUXV4TW9kZWwnXS5tb2RlbFRvcG9sb2d5VHlwZSlcbiAgICAgICAgICAgLnRvRXF1YWwoc2F2ZVJlc3VsdDEubW9kZWxBcnRpZmFjdHNJbmZvLm1vZGVsVG9wb2xvZ3lUeXBlKTtcbiAgICAgICBleHBlY3Qob3V0WydRdXhNb2RlbCddLm1vZGVsVG9wb2xvZ3lCeXRlcylcbiAgICAgICAgICAgLnRvRXF1YWwoc2F2ZVJlc3VsdDEubW9kZWxBcnRpZmFjdHNJbmZvLm1vZGVsVG9wb2xvZ3lCeXRlcyk7XG4gICAgICAgZXhwZWN0KG91dFsnUXV4TW9kZWwnXS53ZWlnaHRTcGVjc0J5dGVzKVxuICAgICAgICAgICAudG9FcXVhbChzYXZlUmVzdWx0MS5tb2RlbEFydGlmYWN0c0luZm8ud2VpZ2h0U3BlY3NCeXRlcyk7XG4gICAgICAgZXhwZWN0KG91dFsnUXV4TW9kZWwnXS53ZWlnaHREYXRhQnl0ZXMpXG4gICAgICAgICAgIC50b0VxdWFsKHNhdmVSZXN1bHQxLm1vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHREYXRhQnl0ZXMpO1xuICAgICAgIGV4cGVjdChvdXRbJ3JlcGVhdC9RdXhNb2RlbCddLm1vZGVsVG9wb2xvZ3lUeXBlKVxuICAgICAgICAgICAudG9FcXVhbChzYXZlUmVzdWx0Mi5tb2RlbEFydGlmYWN0c0luZm8ubW9kZWxUb3BvbG9neVR5cGUpO1xuICAgICAgIGV4cGVjdChvdXRbJ3JlcGVhdC9RdXhNb2RlbCddLm1vZGVsVG9wb2xvZ3lCeXRlcylcbiAgICAgICAgICAgLnRvRXF1YWwoc2F2ZVJlc3VsdDIubW9kZWxBcnRpZmFjdHNJbmZvLm1vZGVsVG9wb2xvZ3lCeXRlcyk7XG4gICAgICAgZXhwZWN0KG91dFsncmVwZWF0L1F1eE1vZGVsJ10ud2VpZ2h0U3BlY3NCeXRlcylcbiAgICAgICAgICAgLnRvRXF1YWwoc2F2ZVJlc3VsdDIubW9kZWxBcnRpZmFjdHNJbmZvLndlaWdodFNwZWNzQnl0ZXMpO1xuICAgICAgIGV4cGVjdChvdXRbJ3JlcGVhdC9RdXhNb2RlbCddLndlaWdodERhdGFCeXRlcylcbiAgICAgICAgICAgLnRvRXF1YWwoc2F2ZVJlc3VsdDIubW9kZWxBcnRpZmFjdHNJbmZvLndlaWdodERhdGFCeXRlcyk7XG4gICAgIH0pKTtcblxuICBpdCgnTWFuYWdlcjogU3VjY2Vzc2Z1bCBkZWxldGVNb2RlbCcsIHJ1bldpdGhMb2NrKGFzeW5jICgpID0+IHtcbiAgICAgICAvLyBGaXJzdCwgc2F2ZSBhIG1vZGVsLlxuICAgICAgIGNvbnN0IGhhbmRsZXIxID0gdGYuaW8uZ2V0U2F2ZUhhbmRsZXJzKCdsb2NhbHN0b3JhZ2U6Ly9RdXhNb2RlbCcpWzBdO1xuICAgICAgIGF3YWl0IGhhbmRsZXIxLnNhdmUoYXJ0aWZhY3RzMSk7XG5cbiAgICAgICAvLyBUaGVuLCBzYXZlIHRoZSBtb2RlbCB1bmRlciBhbm90aGVyIHBhdGguXG4gICAgICAgY29uc3QgaGFuZGxlcjIgPVxuICAgICAgICAgICB0Zi5pby5nZXRTYXZlSGFuZGxlcnMoJ2xvY2Fsc3RvcmFnZTovL3JlcGVhdC9RdXhNb2RlbCcpWzBdO1xuICAgICAgIGF3YWl0IGhhbmRsZXIyLnNhdmUoYXJ0aWZhY3RzMSk7XG5cbiAgICAgICAvLyBBZnRlciBzdWNjZXNzZnVsIHNhdmluZywgZGVsZXRlIHRoZSBmaXJzdCBzYXZlLCBhbmQgdGhlblxuICAgICAgIC8vIGBsaXN0TW9kZWxgIHNob3VsZCBnaXZlIG9ubHkgb25lIHJlc3VsdC5cbiAgICAgICBjb25zdCBtYW5hZ2VyID0gbmV3IEJyb3dzZXJMb2NhbFN0b3JhZ2VNYW5hZ2VyKCk7XG4gICAgICAgYXdhaXQgbWFuYWdlci5yZW1vdmVNb2RlbCgnUXV4TW9kZWwnKTtcbiAgICAgICBjb25zdCBvdXQgPSBhd2FpdCBtYW5hZ2VyLmxpc3RNb2RlbHMoKTtcbiAgICAgICBleHBlY3QoT2JqZWN0LmtleXMob3V0KSkudG9FcXVhbChbJ3JlcGVhdC9RdXhNb2RlbCddKTtcbiAgICAgfSkpO1xufSk7XG4iXX0=