pipeline {
  agent {label 'rhel7'}



  stages {
    stage('Init') {
      steps {
        lastChanges(
          since: 'LAST_SUCCESSFUL_BUILD',
          format:'SIDE',
          matching: 'LINE'
        )
      }
    }


    stage('virtualenv') {
      steps {
        sh '''
          ./createEnv lassi-env
        '''
      }
    }

    stage('test') {
      steps {
        sh '''
          cp settings.py.default settings.py
          source lassi-env/bin/activate
          nosetests --verbose --with-xunit --nocapture
        '''
        junit '*.xml'
      }
    }

  post {
    regression {
        script { env.CHANGED = true }
    }

    fixed {
        script { env.CHANGED = true }
    }

    cleanup {
        do_notify()
    }  
  }
}
