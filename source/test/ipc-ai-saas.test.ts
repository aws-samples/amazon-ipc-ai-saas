import { expect as expectCDK, matchTemplate, MatchStyle } from '@aws-cdk/assert';
import * as cdk from '@aws-cdk/core';
import * as IpcAiSaas from '../lib/ipc-ai-saas-stack';

test('Empty Stack', () => {
    const app = new cdk.App();
    // WHEN
    const stack = new IpcAiSaas.IpcAiSaasStack(app, 'MyTestStack');
    // THEN
    // expectCDK(stack).to(matchTemplate({
    //   "Resources": {}
    // }, MatchStyle.EXACT))
});
